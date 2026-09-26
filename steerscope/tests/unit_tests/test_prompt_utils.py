import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from steerscope.utils.prompt_utils import get_concept_genres


class TestPromptUtils:
    """
    Unit tests for the prompt_utils module
    
    p.s. I am not sure how useful they are, as LLM-based API calls are hard to test.
    I think looking at the raw generation is the best way to go.

    So, I stop prompting cursor to generate more tests after all these. Bye!
    """
    def setup_method(self):
        # Create mock client
        self.mock_client = MagicMock()
        
        # Mock concepts to test
        self.test_concepts = ["math", "programming"]
        
        # Mock API responses for different concepts
        self.mock_responses = [
            "text, math",  # Response for "math"
            "text, code"   # Response for "programming"
        ]

    @patch('steerscope.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    def test_get_concept_genres(self):
        """Test get_concept_genres function"""
        # Configure mock client's chat_completions method
        self.mock_client.chat_completions = AsyncMock(
            return_value=self.mock_responses
        )
        
        # Call the function
        result = asyncio.run(
            get_concept_genres(
                client=self.mock_client,
                concepts=self.test_concepts,
                api_tag="test",
            )
        )
        
        # Verify the client was called correctly
        expected_prompts = [
            "mock_prompt_math",
            "mock_prompt_programming"
        ]
        self.mock_client.chat_completions.assert_awaited_once_with(
            "test.get_concept_genre",
            expected_prompts
        )
        
        # Verify the results
        assert result["math"] == ["text", "math"]
        assert result["programming"] == ["text", "code"]

    @patch('steerscope.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    def test_get_concept_genres_none_response(self):
        """Test get_concept_genres function when LLM responds with 'none'"""
        # Configure mock client with 'none' response
        self.mock_client.chat_completions = AsyncMock(
            return_value=["none", "none"]
        )
        
        # Call the function
        result = asyncio.run(
            get_concept_genres(
                client=self.mock_client,
                concepts=self.test_concepts,
                api_tag="test",
            )
        )
        
        # Verify results default to ["text"] when response is "none"
        assert result["math"] == ["text"]
        assert result["programming"] == ["text"]

    @patch('steerscope.utils.prompt_utils.T_DETERMINE_GENRE', "mock_prompt_{CONCEPT}")
    def test_get_concept_genres_empty_concepts(self):
        """Test get_concept_genres function with empty concepts list"""
        self.mock_client.chat_completions = AsyncMock()
        result = asyncio.run(
            get_concept_genres(
                client=self.mock_client,
                concepts=[],
                api_tag="test",
            )
        )
        
        # Verify empty dict is returned for empty concepts list
        assert result == {}
        # Verify client wasn't called
        self.mock_client.chat_completions.assert_not_awaited()
