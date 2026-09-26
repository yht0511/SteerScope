import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import os
from pathlib import Path
import shutil
import torch
from datasets import Dataset
from steerscope.utils.dataset import DatasetFactory
import pandas as pd
from steerscope.utils.constants import EXAMPLE_TAG, EMPTY_CONCEPT

class TestDatasetFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test cache directory
        cls.cache_dir = Path(__file__).parent / "cache"
        cls.cache_dir.mkdir(exist_ok=True)
        
        # Create master data directory
        cls.master_data_dir = cls.cache_dir / "master_data"
        cls.master_data_dir.mkdir(exist_ok=True)

    def setUp(self):
        # Create mock model, client, and tokenizer
        self.mock_model = MagicMock()
        self.mock_client = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Create mock datasets with more examples
        self.mock_seed_sentences = {
            "text_train": Dataset.from_dict({
                "input": [f"This is test sentence {i}" for i in range(20)]
            }),
            "text_test": Dataset.from_dict({
                "input": [f"This is test sentence {i}" for i in range(20)]
            }),
            "math_train": Dataset.from_dict({
                "input": [f"{i} + {i} = {2*i}" for i in range(20)]
            }),
            "math_test": Dataset.from_dict({
                "input": [f"{i} + {i} = {2*i}" for i in range(20)]
            }),
            "code_train": Dataset.from_dict({
                "input": [f"def test{i}(): pass" for i in range(20)]
            }),
            "code_test": Dataset.from_dict({
                "input": [f"def test{i}(): pass" for i in range(20)]
            })
        }
        
        self.mock_seed_instructions = {
            "text_train": Dataset.from_dict({
                "input": [f"Write about topic {i}" for i in range(20)]
            }),
            "text_test": Dataset.from_dict({
                "input": [f"Write about topic {i}" for i in range(20)]
            }),
            "math_train": Dataset.from_dict({
                "input": [f"Solve problem {i}" for i in range(20)]
            }),
            "math_test": Dataset.from_dict({
                "input": [f"Solve problem {i}" for i in range(20)]
            }),
            "code_train": Dataset.from_dict({
                "input": [f"Write function {i}" for i in range(20)]
            }),
            "code_test": Dataset.from_dict({
                "input": [f"Write function {i}" for i in range(20)]
            })
        }

        # Mock dataset loading
        with patch('steerscope.utils.dataset.load_from_disk') as mock_load:
            mock_load.side_effect = [
                self.mock_seed_sentences,
                self.mock_seed_instructions
            ]
            
            # Initialize DatasetFactory
            self.dataset_factory = DatasetFactory(
                model=self.mock_model,
                client=self.mock_client,
                tokenizer=self.mock_tokenizer,
                dataset_category="instruction",
                num_of_examples=10,  # We now have enough examples (20) to sample 10
                output_length=32,
                dump_dir=str(self.cache_dir),
                master_data_dir=str(self.master_data_dir),
                use_cache=True,
                seed=42,
                api_concurrency=10,
            )

    def test_initialization(self):
        """Test if DatasetFactory initializes correctly"""
        # Check if basic attributes are set correctly
        self.assertEqual(self.dataset_factory.model, self.mock_model)
        self.assertEqual(self.dataset_factory.tokenizer, self.mock_tokenizer)
        self.assertEqual(self.dataset_factory.dataset_category, "instruction")
        self.assertEqual(self.dataset_factory.seed, 42)
        self.assertEqual(self.dataset_factory.api_concurrency, 10)
        self.assertEqual(self.dataset_factory.lm_model.api_batch_size, 5)
        self.assertFalse(hasattr(self.dataset_factory, "negative_df"))
        
        # Check if datasets were loaded
        self.assertEqual(
            len(self.dataset_factory.seed_sentences), 
            len(self.mock_seed_sentences)
        )
        self.assertEqual(
            len(self.dataset_factory.seed_instructions), 
            len(self.mock_seed_instructions)
        )

    def test_api_concurrency_must_split_evenly_across_pair_branches(self):
        with self.assertRaisesRegex(ValueError, "even integer of at least 2"):
            DatasetFactory(
                model=self.mock_model,
                client=self.mock_client,
                tokenizer=self.mock_tokenizer,
                dataset_category="instruction",
                num_of_examples=10,
                output_length=32,
                dump_dir=str(self.cache_dir),
                master_data_dir=str(self.master_data_dir),
                api_concurrency=3,
            )

    def test_prepare_concepts(self):
        """Test prepare_concepts method"""
        # Test case 1: When overwrite_inference_data_dir is set and exists
        with patch('os.path.exists') as mock_exists:
            # Set up the mock to return True for overwrite_inference_data_dir
            mock_exists.return_value = True
            
            # Set overwrite_inference_data_dir
            self.dataset_factory.overwrite_inference_data_dir = "some/path"
            
            # Call prepare_concepts
            concept_genres_map, contrast_concepts_map = self.dataset_factory.prepare_concepts(
                concepts=["concept1", "concept2"]
            )
            
            # Verify empty maps are returned when using pre-generated metadata
            self.assertEqual(concept_genres_map, {})
            self.assertEqual(contrast_concepts_map, {})
            
        # Test case 2: Normal operation (when not using pre-generated metadata)
        self.dataset_factory.overwrite_inference_data_dir = None
        
        # Mock the async functions that get called
        with patch(
            'steerscope.utils.dataset.get_contrast_concepts',
            new_callable=MagicMock,
        ) as mock_contrast, patch(
            'steerscope.utils.dataset.get_concept_genres',
            new_callable=MagicMock,
        ) as mock_genres, \
             patch('steerscope.utils.dataset.run_tasks') as mock_run_tasks:
            
            # run_tasks is mocked, so use inert task sentinels.
            mock_contrast.return_value = object()
            mock_genres.return_value = object()
            
            # Mock run_tasks to return the resolved values
            mock_run_tasks.return_value = [
                {"concept1": ["contrast1"], "concept2": ["contrast2"]},
                {"concept1": ["text"], "concept2": ["code"]}
            ]
            
            # Call prepare_concepts
            concept_genres_map, contrast_concepts_map = self.dataset_factory.prepare_concepts(
                concepts=["concept1", "concept2"]
            )
            
            # Verify the calls were made
            mock_contrast.assert_called_once()
            mock_genres.assert_called_once()
            mock_run_tasks.assert_called_once()
            
            # Verify the returned maps
            self.assertEqual(
                contrast_concepts_map, 
                {"concept1": ["contrast1"], "concept2": ["contrast2"]}
            )
            self.assertEqual(
                concept_genres_map, 
                {"concept1": ["text"], "concept2": ["code"]}
            )

    def test_create_imbalance_eval_df(self):
        """Test create_imbalance_eval_df method"""
        # Create mock pregenerated inference data
        mock_inference_data = pd.DataFrame({
            'input': [f'input_{i}' for i in range(1000)],
            'output': [f'output_{i}' for i in range(1000)],
            'output_concept': ['EEEEE'] * 1000,
            'concept_genre': ['text'] * 1000,
            'category': ['negative'] * 1000,
            'dataset_category': ['instruction'] * 1000
        })

        # Set up the DatasetFactory with pregenerated data
        self.dataset_factory.overwrite_inference_data_dir = "mock/path"
        self.dataset_factory.pregenerated_inference_df = mock_inference_data

        # Test with different subset sizes
        test_cases = [1, 5, 10]  # Different subset_n values to test
        
        for subset_n in test_cases:
            with self.subTest(subset_n=subset_n):
                # Calculate expected number of samples
                expected_samples = subset_n * 100  # As per the code: negative_n_upsamples = int(subset_n*100)
                
                # Get the imbalanced eval dataframe
                result_df = self.dataset_factory.create_imbalance_eval_df(subset_n)
                
                # Verify the results
                self.assertEqual(len(result_df), expected_samples)
                self.assertTrue(all(result_df['category'] == 'negative'))
                self.assertTrue(all(result_df['output_concept'] == EMPTY_CONCEPT))  # EMPTY_CONCEPT
                
                # Verify the DataFrame has all required columns
                required_columns = [
                    'input', 'output', 'output_concept', 
                    'concept_genre', 'category', 'dataset_category'
                ]
                for col in required_columns:
                    self.assertIn(col, result_df.columns)

    def test_create_train_df_generates_same_prompt_pairs(self):
        with patch(
            "steerscope.utils.dataset.response_with_concept",
            new_callable=AsyncMock,
            return_value=["positive-0", "positive-1", "positive-2"],
        ) as with_concept, patch(
            "steerscope.utils.dataset.response_without_concept",
            new_callable=AsyncMock,
            return_value=["negative-0", "negative-1", "negative-2"],
        ) as without_concept:
            result = self.dataset_factory.create_train_df(
                "target concept",
                6,
                {"target concept": ["text"]},
                output_length=32,
            )

        positive = result[result["category"] == "positive"].reset_index(drop=True)
        negative = result[result["category"] == "negative"].reset_index(drop=True)
        self.assertEqual(len(result), 6)
        self.assertEqual(positive["pair_id"].tolist(), [0, 1, 2])
        self.assertEqual(negative["pair_id"].tolist(), [0, 1, 2])
        self.assertEqual(positive["input"].tolist(), negative["input"].tolist())
        self.assertTrue((positive["output_concept"] == "target concept").all())
        self.assertTrue((negative["output_concept"] == EMPTY_CONCEPT).all())
        self.assertEqual(
            with_concept.call_args.kwargs["content"],
            without_concept.call_args.kwargs["content"],
        )

    def test_create_dpo_df_generates_losing_outputs_via_api(self):
        existing = pd.DataFrame({
            "input": ["prompt-0", "prompt-1", "prompt-0", "prompt-1"],
            "winning_output": ["win-0", "win-1", None, None],
            "output_concept": [
                "target concept", "target concept", EMPTY_CONCEPT, EMPTY_CONCEPT,
            ],
            "category": ["positive", "positive", "negative", "negative"],
        })
        with patch(
            "steerscope.utils.dataset.response_without_concept",
            new_callable=AsyncMock,
            return_value=["lose-0", "lose-1"],
        ) as without_concept:
            result = self.dataset_factory.create_dpo_df(
                existing, output_length=24, steer_data_type="concept"
            )

        self.assertEqual(result["category"].tolist(), ["positive", "positive"])
        self.assertEqual(result["losing_output"].tolist(), ["lose-0", "lose-1"])
        self.assertEqual(without_concept.call_args.kwargs["content"], [
            "prompt-0", "prompt-1",
        ])
        self.assertEqual(without_concept.call_args.kwargs["length"], 24)

    def test_create_train_df_rejects_incomplete_api_pairs(self):
        with patch(
            "steerscope.utils.dataset.response_with_concept",
            new_callable=AsyncMock,
            return_value=["positive-0", "positive-1", "positive-2"],
        ), patch(
            "steerscope.utils.dataset.response_without_concept",
            new_callable=AsyncMock,
            return_value=["negative-0", "negative-1"],
        ):
            with self.assertRaisesRegex(ValueError, "incomplete training pairs"):
                self.dataset_factory.create_train_df(
                    "target concept",
                    6,
                    {"target concept": ["text"]},
                    output_length=32,
                )

    def tearDown(self):
        """Clean up after each test"""
        torch.cuda.empty_cache()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove cache directory and its contents
        shutil.rmtree(cls.cache_dir)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()
