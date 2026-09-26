import pandas as pd
import numpy as np
import re
import json
from .alpaca import AlpacaEvaluator
import logging


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class RuleEvaluator(AlpacaEvaluator):
    """
    Evaluator that checks responses against predefined rules without using an LM judge.
    For example, checking if a response contains emojis, follows specific formats, etc.
    """
    
    def __init__(self, node, context, **params):
        super().__init__(node, context, **params)
        self.nlp = params.get("nlp")
        
        # Map rule types to their evaluation functions
        self.rule_functions = {
            "emoji": self._check_emoji,
            "length": self._check_length,
            "keyword": self._check_keyword,
            "code": self._check_code,
            "question": self._check_question,
            "all_capital": self._check_uppercase,
            "chinese": self._check_chinese,
            "spanish": self._check_spanish,
            "all_lowercase": self._check_all_lowercase,
            "korean": self._check_korean,
            "word_limit_max": self._check_word_limit_max,
            "word_limit_min": self._check_word_limit_min,
            "postscript": self._check_postscript,
            "json_format": self._check_json_format,
            "numbered_list": self._check_numbered_list,
            "double_breaks": self._check_double_breaks,
            "asterisk_sep": self._check_asterisk_separation,
            "start_phrase": self._check_starts_with_phrase,
            "quoted_words": self._check_words_in_quotes,
            "max_sentences": self._check_max_sentences,
            "help_phrase": self._check_ends_with_help,
            "exclamation": self._check_has_exclamation,
            "hashtags": self._check_has_hashtags,
            "citations": self._check_has_citations,
            "same_letter_ending": self._check_same_letter_ending,
            "telephone_number": self._check_telephone_number,
            "email": self._check_email,
            "date_format": self._check_date_format,
            "past_tense": self.is_past_tense,
            "passive_voice": self.contains_passive_voice,
        }
        
        # Additional parameters for specific rules
        self.min_length = 50
        self.max_length = 800
        self.keywords = ["beauty", 'pretty']

    def render_report(self, result, output_dir=None):
        return self._render_curve_report(
            result,
            {"rule_following": "Rule following"},
            output_dir,
            columns=1,
            y_limits=(0.0, 2.0),
        )

    def score(self, model, inference):
        from steerscope.utils.constants import CONCEPT_TO_RULE, NEED_STANZA

        self.model_name = model.method
        self.target = model.target
        self.concept_id = model.concept.concept_id
        if model.concept.text in NEED_STANZA and self.nlp is None:
            try:
                import stanza
            except ImportError as error:
                raise RuntimeError(
                    "The selected rule requires stanza and its English model."
                ) from error
            self.nlp = stanza.Pipeline(
                "en", processors="tokenize,pos", device="cpu"
            )
        try:
            rule_type = CONCEPT_TO_RULE[model.concept.text]
        except KeyError as error:
            raise KeyError(
                f"No rule mapping is configured for concept '{model.concept.text}'."
            ) from error
        result = self.compute_metrics(inference, rule_type=rule_type)
        return self._metric_rows(model, result, inference=inference), self._sample_rows(
            model, inference, result
        )
        
    def __str__(self): 
        return f'RuleEvaluator'
    
    def _check_emoji(self, text):
        """Check if text contains emojis"""
        try:
            import emoji
        except ImportError as error:
            raise RuntimeError("The emoji rule requires the 'emoji' package.") from error
        emoji_count = sum(1 for char in text if char in emoji.EMOJI_DATA)
        # Return a score between 0 and 2 based on emoji presence
        if emoji_count > 2:
            return 2.0
        elif emoji_count > 0:
            return 1.0
        return 0.0
    
    def _check_length(self, text):
        """Check if text length is within specified range"""
        length = len(text)
        if self.min_length <= length <= self.max_length:
            return 2.0
        elif length < self.min_length:
            return length / self.min_length * 2.0
        else:
            return max(0.0, 2.0 - (length - self.max_length) / self.max_length)
    
    def _check_keyword(self, text):
        """Check if text contains specified keywords"""
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in self.keywords if keyword.lower() in text_lower)
        if not self.keywords:
            return 0.0
        return min(2.0, (found_keywords / len(self.keywords)) * 2.0)
    
    def _check_code(self, text):
        """Check if text contains code blocks"""
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        if code_blocks:
            return 2.0
        elif re.search(r'`[^`]+`', text):  # Inline code
            return 1.0
        return 0.0
    
    def _check_question(self, text):
        """Check if text contains questions"""
        questions = re.findall(r'\?', text)
        if len(questions) > 2:
            return 2.0
        elif len(questions) > 0:
            return 1.0
        return 0.0

    def _check_uppercase(self, text):
        """Check if text contains uppercase words for emphasis"""
        text = text.replace("<end_of_turn>", "")
        if not text:
            return 0.0
        words = text.split()
        uppercase_words = [word for word in words if word.isupper()]
        try:
            percentage = (len(uppercase_words) / len(words)) * 2
        except:
            percentage = 0.0
        
        return percentage
    
    def contains_passive_voice(self,text):
        idoc = self.nlp(text)
        for sentence in idoc.sentences:
            for word in sentence.words:
                # Passive voice typically shows up with 'Voice=Pass' in the features
                if word.upos == 'VERB' and word.feats and 'Voice=Pass' in word.feats:
                    return 2.0

        return 0.0
        
    def _check_chinese(self, text):
        """Check if text contains Chinese characters"""
        # Unicode ranges for Chinese characters
        text = text.replace("<end_of_turn>", "")
        try:
            import langdetect
            r = langdetect.detect(text)
            return 2.0 if r == 'zh-cn' else 0.0
        except ImportError as error:
            raise RuntimeError("Language rules require the 'langdetect' package.") from error
        except Exception:
            return 0.0
    
    def _check_spanish(self, text):
        """Check if text is only in Spanish"""
        # Basic Spanish character set check (including ñ, á, é, í, ó, ú, ü)
        text = text.replace("<end_of_turn>", "")
        try:
            import langdetect
            r = langdetect.detect(text)
            return 2.0 if r == 'es' else 0.0
        except ImportError as error:
            raise RuntimeError("Language rules require the 'langdetect' package.") from error
        except Exception:
            return 0.0
    
    def _check_all_caps(self, text):
        """Check if text uses only capital letters"""
        text = text.replace("<end_of_turn>", "")
        if not text.strip():
            return 0.0
        return 2.0 if text.isupper() else 0.0

    def _check_all_lowercase(self, text):
        """Check if all words are in lowercase letters"""
        text = text.replace("<end_of_turn>", "").strip()
        if not text:
            return 0.0
        words = text.split()
        lowercase_words = [word for word in words if word.islower()]
        percentage = (len(lowercase_words) / len(words))
        return percentage * 2

    def _check_korean(self, text):
        """Check if text is only in Korean"""
        korean_pattern = re.compile(r'^[\uac00-\ud7a3\s.,!?]+$')
        return 2.0 if bool(korean_pattern.match(text.strip())) else 0.0

    def _check_word_limit_max(self, text, max_words=50):
        tokens = re.findall(r"\w+", text)
        num_words = len(tokens)     
        return 2.0 if num_words <= max_words else 0.0

    def _check_word_limit_min(self, text, min_words=800):
        tokens = re.findall(r"\w+", text)
        num_words = len(tokens)     
        return 2.0 if num_words >= min_words else 0.0

    def _check_postscript(self, text):
        """Check if text includes a P.S. at the end"""
        text = text.replace("<end_of_turn>", "")
        return 2.0 if bool(re.search(r'P\.S\..*$', text, re.MULTILINE)) else 0.0

    def _check_json_format(self, value):
        """Check if text is in valid JSON format"""
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip())
        try:
            json.loads(value)
        except ValueError as _:
            return False
        return True

    def _check_numbered_list(self, text):
        """Check if text contains any numbered list item (e.g., 1., 2., 3., ... n.)"""
        # Match any number followed by a period and a space or end of string
        if re.search(r'\b\d+\.', text):
            return 2.0
        return 0.0

    def _check_double_breaks(self, text):
        """Check if paragraphs are separated by double line breaks"""
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return 2.0
        return 0.0

    def _check_asterisk_separation(self, text):
        """Check if paragraphs are separated by ***"""
        return 2.0 if '***' in text else 0.0

    def _check_starts_with_phrase(self, text, phrase="Here is my response"):
        """Check if text starts with specific phrase"""
        text = text.replace("<end_of_turn>", "")
        if text.strip().startswith(phrase):
            return 2.0
        elif "Here is my response" in text.strip():
            return 1.0
        return 0.0

    def _check_words_in_quotes(self, text):
        """Check if every word is wrapped in double quotation marks"""
        words = text.split()
        if not words:
            return 0.0
        text = text.replace("<end_of_turn>", "")
        quoted_words = sum(1 for word in words if word.startswith('"') and word.endswith('"'))
        return (quoted_words / len(words)) * 2.0

    def _check_max_sentences(self, text, max_sentences=3):
        """Check if text contains exactly max_sentences sentences, handling special cases."""
        # Pre-process text to handle common abbreviations and initials
        text = re.sub(r'(?<=[A-Z])\.\s*(?=[A-Z]\.)', 'DOT', text)  # Handle initials like "H.S."
        text = re.sub(r'Mr\.|Mrs\.|Dr\.|Ms\.|U\.S\.|D\.|Jr\.|Sr\.', lambda m: m.group().replace('.', 'DOT'), text)
        
        # Split into sentences and clean up
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Restore periods and recount
        sentences = [s.replace('DOT', '.') for s in sentences]
        
        # Return score based on exact match of sentence count
        return 2.0 if len(sentences) == max_sentences else 0.0

    def _check_ends_with_help(self, text):
        """Check if text ends with help phrase"""
        text = text.replace("<end_of_turn>", "")
        if text.strip().endswith("Is there anything else I can help with?"):
            return 2.0
        elif "Is there anything else I can help with" in text.strip():
            return 1.0
        return 0.0



    def is_past_tense(self, word):
        doc = self.nlp(word)
        for sentence in doc.sentences:
            for word in sentence.words:
                # In Universal POS tags, 'VERB' + 'Tense=Past' indicates a past tense verb
                if word.upos == 'VERB' and 'Tense=Past' in word.feats if word.feats else '':
                    return 2.0
        return 0.0
    
    def _check_has_exclamation(self, text):
        """Check if text contains exclamation marks"""
        text = text.replace("<end_of_turn>", "")
        exclamation_count = text.count('!')
        return min(2.0, exclamation_count * 0.5)

    def _check_has_hashtags(self, text, min_hashtags=4):
        """Check if text includes at least min_hashtags hashtags"""
        hashtags = re.findall(r'#\w+', text)
        if len(hashtags) >= min_hashtags:
            return 2.0
        return (len(hashtags) / min_hashtags) * 2.0

    def _check_has_citations(self, text):
        """Check if text includes citations with URLs"""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        urls = url_pattern.findall(text)
        return 2.0 if urls else 0.0

    def _check_same_letter_ending(self, text):
        """Check if every sentence ends with the letter 'y'"""
        # Split into sentences and clean up
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0
        
        # Get last letter of each sentence (ignoring punctuation)
        last_letters = [re.sub(r'[.!?,\s]', '', s)[-1].lower() for s in sentences]
        
        # Check if all letters are 'y'
        return 2.0 if all(letter == 'y' for letter in last_letters) else 0.0

    def _check_telephone_number(self, text):
        """Recognize common local, international, and alphanumeric telephone-number formats."""
        phone_patterns = [
            # Standard US formats
            r'\(\d{3}\)\s*[\d\-\s]+\d{4}',                    # (123) 456-7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',                # 123-456-7890
            
            # International format
            r'\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # +1-123-456-7890
            
            # Local format
            r'\d{3}[-.\s]?\d{4}',                            # 555-1234
            
            # Alphanumeric formats
            r'\(\d{3}\)\s*\d{3}[-.\s][A-Z]+\s*\(\d+\)',     # (212) 555-STAGE (7824)
            #r'\d{3}[-.\s][A-Z]+',                           # 555-STAGE
            #r'\d+[-.\s][A-Z]+',                             # 1-800-FLOWERS
            
            # Additional formats with letters
            r'\(\d{3}\)\s*\d{3}[-.\s][A-Z\d]+',            # (212) 555-STAGE
            r'\d{3}[-.\s]\d{3}[-.\s][A-Z\d]+',             # 212-555-STAGE
        ]
        
        # Count unique phone numbers found
        phone_numbers = set()
        for pattern in phone_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            phone_numbers.update(match.group() for match in matches)
        
        # Return 2.0 if there are at least 2 unique phone numbers
        return 2.0 if len(phone_numbers) >= 1 else 0.0

    def _check_email(self, text):
        """Check if text contains an email address"""
        text = text.replace("<end_of_turn>", "")
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return 2.0 if re.search(email_pattern, text) else 0.0

    def _check_date_format(self, text):
        """Check if text contains a date in YYYY-MM-DD format"""
        date_pattern = r'\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b'
        return 2.0 if re.search(date_pattern, text) else 0.0
    
    def _evaluate_text(self, text):
        """Apply the selected rule to evaluate the text"""
        if self.rule_type in self.rule_functions:
            return self.rule_functions[self.rule_type](text)
        else:
            logger.warning(f"Unknown rule type: {self.rule_type}")
            return 0.0
    
    def compute_metrics(self, data, write_to_dir=None, rule_type=None):
        """Evaluate generated responses with the configured deterministic rule."""
        # Use the rule_type passed to this method, if provided
        current_rule_type = rule_type
        
        if current_rule_type not in self.rule_functions:
            raise ValueError(f"Unknown rule type: {current_rule_type}")
            
        logger.warning(
            f"Starting rule evaluation for concept_id: {self.concept_id}, "
            f"rule_type: {current_rule_type}, model: {self.model_name}")
        
        data_copy = data.copy()
        
        # Apply rule evaluation to each row
        rule_ratings = []
        for idx, row in data_copy.iterrows():
            generation = row[f"{self.model_name}_steered_generation"]
            rating = self.rule_functions[current_rule_type](generation)
            rule_ratings.append(rating)
        
        # Store ratings in the dataframe
        data_copy[f"{self.model_name}_rule_rating"] = rule_ratings
        
        # Group by factor and compute means
        metrics = {
            "rule_following": [],
            "factor": [],
            "raw_rule_following": rule_ratings  # Add raw ratings similar to LMJudgeEvaluator
        }
        
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            metrics["rule_following"].append(group[f"{self.model_name}_rule_rating"].mean())
            metrics["factor"].append(factor)
        
        return metrics
    
    
    def compute_metrics_train(self, data, write_to_dir=None, rule_type=None):
        """Evaluate training responses with the configured deterministic rule."""
        # Use the rule_type passed to this method, if provided
        current_rule_type = rule_type
        
        if current_rule_type not in self.rule_functions:
            raise ValueError(f"Unknown rule type: {current_rule_type}")
            
        logger.warning(
            f"Starting rule evaluation for concept_id: {self.concept_id}, "
            f"rule_type: {current_rule_type}, model: {self.model_name}")
        
        data_copy = data.copy()
        
        # Apply rule evaluation to each row
        rule_ratings_winning = []
        rule_ratings_losing = []
        for idx, row in data_copy.iterrows():
            generation = row["winning_output"]
            rating = self.rule_functions[current_rule_type](generation)
            rule_ratings_winning.append(rating)
            generation_losing = row["losing_output"]
            rating = self.rule_functions[current_rule_type](generation_losing)
            rule_ratings_losing.append(rating)

        
        # Store ratings in the dataframe
        data_copy[f"winning_rule_rating"] = rule_ratings_winning
        data_copy[f"losing_rule_rating"] = rule_ratings_losing
        
        # Group by factor and compute means
        metrics = {
            "rule_following": [],
            "factor": [],
            "raw_rule_following_winning": rule_ratings_winning,  # Add raw ratings similar to LMJudgeEvaluator
            "raw_rule_following_losing": rule_ratings_losing
        }
        
        return metrics
