import json
import os
import re
import time
import torch
from loguru import logger
from pathlib import Path
from neurips25.eval import *
from neurips25.utils.confidence import build_confidence_record, extract_choice_letters
from neurips25.utils.question_utils import detect_outcome_question_type, load_hancock_patient_data
from neurips25.utils.staging import stage_patient
from neurips25.utils.survival_lookup import (
    format_population_context,
    lookup_population_stats,
    population_informed_answer,
)

class DoctorAgent:
    def __init__(
        self,
        main_llm,
        oracle_llm,
        model_name,
        output_dir="./agent_logs",
        cases_path="data/hancock/cases",
        inject_population_stats=True,
    ):
        """
        Initialize the PatientAgent with the main LLM and oracle LLM for answers.
        """
        self.main_llm = main_llm
        self.oracle_llm = oracle_llm
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cases_path = cases_path
        self.inject_population_stats = inject_population_stats
        self.chat_history = []
        self.strike_count = 0

    def _is_text_only_model(self):
        name = (self.model_name or "").lower()
        return any(tag in name for tag in ("meditron", "llama31", "llama-3", "qwen3"))

    @staticmethod
    def _extract_valid_answer(response):
        matches = re.findall(r"\[ANSWER:\s*([^\]]+)\]", response)
        if not matches:
            return None
        ans = matches[-1].strip()
        if (len(ans) >= 2 and ans[0].lower() in "abcdef" and ans[1] in ")] ") or (
            len(ans) == 1 and ans[0].lower() in "abcdef"
        ):
            return ans
        return None

    @staticmethod
    def _parse_requested_files(response):
        requested = re.findall(r"\[REQUEST:\s*([^\]]+)\]", response)
        if len(requested) == 1 and requested[0].startswith("[REQUEST:") and "," in requested[0]:
            requested = requested[0].split(",")
        return [f.strip() for f in requested if f.strip()]

    def _build_system_message(self):
        system_message = (
            "You are a pathologist AI assistant expert at analyzing patient data and answering user questions.\n"
            "You will be provided with files that you are allowed to read.\n"
            "More files may become available as the conversation progresses.\n"
            "To ask for files, include in your reasoning [REQUEST: filename.extension] for each file you need. "
            "Example: [REQUEST: image1.jpg] [REQUEST: image2.jpg]\n"
            "You may request multiple files at once if necessary. If you ask for a file wait to receive it from the user.\n"
            "To provide a final answer to a question, include [ANSWER: CHOICE) your answer] in your response, "
            "specifying the answer choice you picked (A, B, C, D, E, or F).\n"
            "Also include your self-reported confidence as [CONFIDENCE: 0.XX] where 0.XX is a number from 0.0 to 1.0 "
            "(e.g. [CONFIDENCE: 0.85] for 85% confidence).\n"
            "Do NOT combine [REQUEST] and [ANSWER] in the same message — either request files OR give a final answer.\n"
            "You MUST ONLY provide [ANSWER] when you have all necessary information."
        )
        if self._is_text_only_model():
            system_message += (
                "\n\nIMPORTANT — text-only mode:\n"
                "- You CANNOT view image files (.jpg, .png); requesting them only marks them as included.\n"
                "- Only .json and .txt files provide readable text content when requested.\n"
                "- Use exact filenames from the available-files list; do not invent file names.\n"
                "- If a file is reported as 'not found', never request it again.\n"
                "- If needed data is unavailable, give your best [ANSWER] with [CONFIDENCE] instead of looping."
            )
        return system_message

    def _available_files_hint(self, basenames):
        if not basenames:
            return ""
        readable = [b for b in basenames if b.lower().endswith((".json", ".txt"))]
        hint = f"\nAvailable files ({len(basenames)}): {', '.join(basenames)}\n"
        if self._is_text_only_model():
            hint += (
                f"Readable text files: {', '.join(readable) if readable else '(none yet)'}\n"
                "Image files cannot be visually inspected in text-only mode.\n"
            )
        return hint

    def _record_answer(self, question, expected_answer, response, full_response, correct,
                       all_files_accessed, all_hallucinated, question_time, conversation,
                       choices, outcome_kwargs):
        self.strike_count = 0
        self.chat_history.append(self._make_log_entry(
            question, expected_answer, response, correct,
            all_files_accessed, all_hallucinated, question_time, conversation,
            choices, full_response=full_response, **outcome_kwargs,
        ))


    def _parse_files(self, file_paths):
        """
        Parse the file paths to separate images and text files.
        """
        images = []
        texts = []
        
        for path in file_paths:
            ext = Path(path).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                images.append(path)
            elif ext in ['.txt', '.json']:
                texts.append(path)

        file_overview = []
        for img in images:
            file_overview.append(f"[FILE: {os.path.basename(img)}]")
        for txt_path in texts:
            file_overview.append(f"[FILE: {os.path.basename(txt_path)}]")

        return file_overview
        

    def _attach_files(self, requested_files, current_file_basenames, current_file_paths):
        """
        Attach files to the conversation based on the requested files.
        """
        file_msg = ""
        attached_files = []
        hallucinated_files = []
        all_given_files = []

        for req_file in set(requested_files):
            req_file = req_file.replace(" ", "") # some models add spaces before extension
            # Check if the requested file is in the current file basenames
            if req_file not in current_file_basenames:
                file_msg += f"[FILE: {req_file}] not found. Only ask for files that were listed to you earlier! Example request format for 2 images: [REQUEST: image1.jpg] [REQUEST: image2.jpg]\n"
                hallucinated_files.append(req_file)
                continue

            # Add to all given files
            all_given_files.append(req_file)
            
            # Check if the requested file is in the current file paths
            full_path = next((fp for fp in current_file_paths if os.path.basename(fp) == req_file), None)
            if full_path:
                # Add image path
                if full_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    attached_files.append(full_path)
                    file_msg += f"[FILE: {req_file}] included\n"
                # Else text file, so read it
                else:
                    try:
                        with open(full_path, 'r') as f:
                            content = json.load(f) if full_path.endswith('.json') else f.read()
                            file_msg += f"[FILE: {req_file}] included\n{content}\n"
                    except Exception as e:
                        file_msg += f"[ERROR: Failed to read {req_file}: {e}]\n"
        return file_msg, attached_files, hallucinated_files, all_given_files
    

    def _dettach_files(self, conversation):
        """
        Detach files from the conversation.
        """
        for entry in conversation:
            if 'files' in entry:
                del entry['files']
            if entry['role'] == 'user':
                content = entry['content']
                file_names = re.findall(r"\[FILE: (.+?)\] included\n", content)
                content = ""
                for file_name in file_names:
                    content += f"[FILE: {file_name}] was accessed by you\n"
                if file_names:
                    entry['content'] = content + "You can access all these files once again if necessary by asking for them in the format [REQUEST: filename.extension].\n"
                    logger.info(f"Files detached from conversation: {file_names}")
                    file_names = []
        return conversation

    def _build_outcome_context(self, case_id, question):
        """Stage patient and build SEER/TCGA population context for survival/recurrence questions."""
        outcome_type = detect_outcome_question_type(question)
        if outcome_type is None:
            return None, None, None, None, None
        if not self.inject_population_stats:
            return outcome_type, None, None, None, None

        patient_data = load_hancock_patient_data(case_id, self.cases_path)
        pathological = patient_data.get("pathological")
        if not pathological:
            logger.warning(f"No pathological data for case {case_id}; skipping population lookup")
            return outcome_type, None, None, None, None

        clinical = patient_data.get("clinical", {})
        staging = stage_patient(pathological, clinical)
        population_stats = lookup_population_stats(staging)
        informed = population_informed_answer(outcome_type, population_stats)
        context = format_population_context(staging, population_stats, outcome_type)
        return outcome_type, staging, population_stats, informed, context

    def _make_log_entry(
        self,
        question,
        expected_answer,
        response,
        correct,
        all_files_accessed_for_question,
        all_halucinated_files,
        question_time,
        conversation,
        choices,
        full_response=None,
        outcome_type=None,
        staging=None,
        population_stats=None,
        population_answer=None,
    ):
        selected_letter = response.strip()[0].upper() if response.strip() else ""
        entry = {
            "question": question,
            "answer": expected_answer,
            "response": response,
            "correct": correct,
            "files_accessed": all_files_accessed_for_question,
            "files_hallucinated": all_halucinated_files,
            "question_time": question_time,
        }
        confidence = build_confidence_record(
            full_response or response, selected_letter, choices, self.main_llm, conversation
        )
        entry.update(confidence)

        if outcome_type:
            entry["outcome_question_type"] = outcome_type
            entry["patient_staging"] = staging
            entry["population_stats"] = population_stats
            entry["population_informed_answer"] = population_answer
            if population_answer:
                entry["population_answer_matches_model"] = (
                    selected_letter == population_answer.get("answer_letter")
                )
        return entry
    
    @torch.no_grad()
    def run_case(self, case_data, case_id="patient_case"):
        """
        Run the agent on a single patient case.
        """
        logger.info(f"Running agent on {case_id}")

        conversation = []
        self.chat_history = []  # Clear previous runs
        self.strike_count = 0

        # System prompt
        conversation.append({"role": "system", "content": self._build_system_message()})

        # Initialize context and file tracking
        context_message = ""
        current_file_paths = []
        current_file_basenames = []

        context_message = ""
        for entry in case_data:
            # Patient context
            if 'context' in entry:
                # Update current context
                context_message = f"You are given the following new patient information: \n{entry['context']}\n"
            
            # Patient files
            elif 'file_paths' in entry:
                # Update available files
                current_file_paths = entry['file_paths']
                current_file_basenames = [os.path.basename(f) for f in current_file_paths]
                file_overview = self._parse_files(current_file_paths)

                # Announce new available files
                context_message += f"New files available:\n" + "\n".join(file_overview) + "Remember that you can ask for files by providing the following tag [REQUEST: filename.extension]. You may also ask for multiple files at once if necessary. If you ask for a file wait to receive it from the user.\n"

            # Patient question
            elif 'question' in entry:
                # Now we can ask a question
                question = entry['question']
                expected_answer = entry['answer']
                choices = extract_choice_letters(question)

                outcome_type = None
                staging = None
                population_stats = None
                population_answer = None
                outcome_result = self._build_outcome_context(case_id, question)
                if outcome_result[0] is not None:
                    outcome_type, staging, population_stats, population_answer, pop_context = outcome_result
                    if pop_context:
                        context_message += pop_context

                # Detach files from the conversation
                conversation = self._dettach_files(conversation)

                logger.info(f"Processing question: {question.strip()}")
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            f"{context_message}\n Question: {question}\n"
                            f"{self._available_files_hint(current_file_basenames)}"
                            "Remember that you can ask for files by providing the following tag "
                            "[REQUEST: filename.extension]. You may also ask for multiple files at once if necessary\n"
                        ),
                    }
                )
                # Empty the context message since we have included it in the conversation
                context_message = ""

                # Start the conversation loop until a valid answer is provided
                question_start_time = time.time()
                prev_requested_files = []
                all_files_accessed_for_question = []
                all_halucinated_files = []
                hallucination_rounds = 0
                turn_count = 0
                max_turns = 10
                outcome_kwargs = dict(
                    outcome_type=outcome_type,
                    staging=staging,
                    population_stats=population_stats,
                    population_answer=population_answer,
                )
                while True:
                    turn_count += 1
                    response = self.main_llm.evaluate(messages=conversation)
                    logger.debug(f"Model response: {response}")
                    conversation.append({"role": "assistant", "content": response})

                    valid_answer = self._extract_valid_answer(response)
                    requested_files = self._parse_requested_files(response)

                    # Accept a valid answer even if the model also requested files in the same turn.
                    if valid_answer:
                        question_end_time = time.time()
                        correct = expected_answer.strip()[0].lower() == valid_answer.strip()[0].lower()
                        self._record_answer(
                            question, expected_answer, valid_answer, response, correct,
                            all_files_accessed_for_question, all_halucinated_files,
                            question_end_time - question_start_time, conversation,
                            choices, outcome_kwargs,
                        )
                        break

                    if requested_files:
                        if requested_files == prev_requested_files:
                            self.strike_count += 1
                            logger.info("Model repeated the same file request.")
                        else:
                            prev_requested_files = requested_files
                            file_msg, attached_files, hallucinated_files, all_given_files = self._attach_files(
                                requested_files, current_file_basenames, current_file_paths,
                            )
                            logger.info(f"Files: {file_msg}")
                            all_files_accessed_for_question.extend(all_given_files)
                            all_halucinated_files.extend(hallucinated_files)
                            conversation.append({"role": "user", "content": file_msg, "files": attached_files})
                            if hallucinated_files and not all_given_files:
                                hallucination_rounds += 1
                            else:
                                hallucination_rounds = 0

                        if hallucination_rounds >= 2 or self.strike_count >= 2:
                            conversation.append({
                                "role": "user",
                                "content": (
                                    "Stop requesting files. The requested files do not exist or were already denied. "
                                    "Do not combine [REQUEST] and [ANSWER]. Provide your final "
                                    "[ANSWER: LETTER) choice] and [CONFIDENCE: 0.XX] now using available information."
                                ),
                            })
                            self.strike_count = 0
                            hallucination_rounds = 0
                    elif turn_count >= max_turns:
                        logger.info(f"Question hit turn limit ({max_turns}); recording as incorrect.")
                        question_end_time = time.time()
                        self._record_answer(
                            question, expected_answer, response[:200], response, False,
                            all_files_accessed_for_question, all_halucinated_files,
                            question_end_time - question_start_time, conversation,
                            choices, outcome_kwargs,
                        )
                        break
                    else:
                        self.strike_count += 1
                        if self.strike_count <= 2:
                            logger.info("Model did not provide a valid answer or request.")
                            conversation.append({
                                "role": "user",
                                "content": (
                                    "Please provide the final answer in [ANSWER: LETTER) answer] specifying the "
                                    "answer choice letter you picked (A, B, C, D, E, or F) or ask for files with "
                                    "[REQUEST: filename.extension]. Do not combine both in one message."
                                ),
                            })
                        else:
                            logger.info("Model failed to provide a valid answer or request after 3 attempts.")
                            question_end_time = time.time()
                            self._record_answer(
                                question, expected_answer, response[:200], response, False,
                                all_files_accessed_for_question, all_halucinated_files,
                                question_end_time - question_start_time, conversation,
                                choices, outcome_kwargs,
                            )
                            break
                        
        # Log the entire conversation
        self.chat_history.append({
            "conversation": conversation
        })
        self._store_log(case_id)
        return self.chat_history


    def _store_log(self, case_id):
        """
        Store the chat history in a JSON file.
        """
        log_path = self.output_dir / f"{case_id}_chatlog_{int(time.time())}.json"
        with open(log_path, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        logger.info(f"Chat history saved to {log_path}")

