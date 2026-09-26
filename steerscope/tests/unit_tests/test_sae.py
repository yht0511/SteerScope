import unittest
import torch
import json
import shutil
from pathlib import Path
from steerscope.models.sae import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch, MagicMock, call
import numpy as np
import os


class TestGemmaScopeSAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create cache directory
        cls.cache_dir = Path(__file__).parent / "cache"
        cls.cache_dir.mkdir(exist_ok=True)
        
        # Create dummy metadata
        cls.test_metadata = [
            {"concept_id": 0, "concept": "test concept 1", 
             "ref": "https://www.neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/0",  # fake link
             "concept_genres_map": {"test concept 1": ["text"]}},
            {"concept_id": 1, "concept": "test concept 2", 
             "ref": "https://www.neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/1",  # fake link
             "concept_genres_map": {"test concept 2": ["code"]}}
        ]
        
        # Write metadata to file
        cls.metadata_path = cls.cache_dir / "metadata.jsonl"
        with open(cls.metadata_path, 'w') as f:
            for item in cls.test_metadata:
                f.write(json.dumps(item) + '\n')
        
    def setUp(self):
        pass

    def test_load_metadata_flatten(self):
        """Test load_metadata_flatten"""
        metadata_flatten = load_metadata_flatten(
            metadata_path=self.metadata_path)
        self.assertEqual(len(metadata_flatten), len(self.test_metadata))
        self.assertEqual(metadata_flatten[0]["concept"], self.test_metadata[0]["concept"])
        self.assertEqual(metadata_flatten[1]["concept"], self.test_metadata[1]["concept"])
        self.assertEqual(metadata_flatten[0]["ref"], self.test_metadata[0]["ref"])
        self.assertEqual(metadata_flatten[1]["ref"], self.test_metadata[1]["ref"])
        self.assertEqual(metadata_flatten[0]["concept_genres_map"], self.test_metadata[0]["concept_genres_map"])
        self.assertEqual(metadata_flatten[1]["concept_genres_map"], self.test_metadata[1]["concept_genres_map"])
        self.assertEqual(metadata_flatten[0]["concept_id"], self.test_metadata[0]["concept_id"])
        self.assertEqual(metadata_flatten[1]["concept_id"], self.test_metadata[1]["concept_id"])

    def test_training_capabilities_match_original_sae_variants(self):
        self.assertFalse(GemmaScopeSAE.requires_training_args)
        self.assertTrue(GemmaScopeSAEMaxAUC.requires_training_args)

    @patch("steerscope.models.sae._neuronpedia_max_activation")
    def test_baseline_prepares_training_calibration_artifact(self, mock_max):
        mock_max.side_effect = [11.0, 22.0]
        (self.cache_dir / "GemmaScopeSAE_scale.pt").unlink(missing_ok=True)

        scale_path = GemmaScopeSAE.prepare_calibration_artifact(
            self.metadata_path,
            self.cache_dir,
            [0, 1],
            master_data_dir=self.cache_dir,
        )

        self.assertEqual(
            torch.load(scale_path, weights_only=True).tolist(),
            [11.0, 22.0],
        )

    @patch("steerscope.models.sae._neuronpedia_max_activation")
    def test_baseline_calibration_preserves_sparse_concept_ids(self, mock_max):
        metadata = [
            {
                "concept_id": concept_id,
                "concept": f"concept {concept_id}",
                "ref": f"https://www.neuronpedia.org/test/source/{concept_id}",
                "concept_genres_map": {f"concept {concept_id}": ["text"]},
            }
            for concept_id in range(7)
        ]
        metadata_path = self.cache_dir / "sparse_metadata.jsonl"
        with open(metadata_path, "w") as file:
            for item in metadata:
                file.write(json.dumps(item) + "\n")
        scale_path = self.cache_dir / "SparseSAE_scale.pt"
        scale_path.unlink(missing_ok=True)
        mock_max.side_effect = [11.0, 66.0]

        GemmaScopeSAE.prepare_calibration_artifact(
            metadata_path,
            self.cache_dir,
            [1, 6],
            master_data_dir=self.cache_dir,
            model_name="SparseSAE",
        )
        scales = torch.load(scale_path, weights_only=True)

        self.assertEqual(scales.shape, (7,))
        self.assertEqual(scales[[1, 6]].tolist(), [11.0, 66.0])
        self.assertTrue(
            torch.equal(scales[[0, 2, 3, 4, 5]], torch.ones(5))
        )

    def test_save_pruned_sae(self):
        """Test save_pruned_sae with mocked API and HF downloads"""
        # Create dummy SAE parameters
        sae_params = {
            'W_dec': np.random.randn(10, 128),  # 10 features, 128 dimensions
            'W_enc': np.random.randn(128, 10),  # Matching encoder weights
            'b_dec': np.random.randn(128),      # Decoder bias
            'b_enc': np.random.randn(10),       # Encoder bias
            'threshold': np.random.randn(10)     # Thresholds for each feature
        }
        
        # Create mock metadata file
        metadata = [
            {
                "concept": "test_concept_1",
                "ref": "https://www.neuronpedia.org/features/test/3", # index 3
                "concept_genres_map": {"test_concept_1": ["genre1"]},
                "concept_id": 0
            },
            {
                "concept": "test_concept_2", 
                "ref": "https://www.neuronpedia.org/features/test/7", # index 7
                "concept_genres_map": {"test_concept_2": ["genre2"]},
                "concept_id": 1
            }
        ]

        metadata_path = self.cache_dir / "test_metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for m in metadata:
                f.write(json.dumps(m) + '\n')

        # Mock API response
        mock_api_response = {
            "source": {
                "hfRepoId": "test/repo",
                "hfFolderId": "test_folder"
            }
        }

        # Set up all our mocks
        with patch('requests.get') as mock_get, \
            patch('steerscope.models.sae.hf_hub_download') as mock_hf_download, \
            patch.dict(os.environ, {'NP_API_KEY': 'fake_key'}):
            
            # Configure mock API response
            mock_get.return_value.json.return_value = mock_api_response
            
            # Mock HF download to return our dummy params
            mock_params_path = self.cache_dir / "mock_params.npz"
            np.savez(mock_params_path, **sae_params)
            mock_hf_download.return_value = str(mock_params_path)
            
            # Call the function
            saved_params = save_pruned_sae(
                metadata_path=metadata_path,
                dump_dir=self.cache_dir,
                savefile="test_sae.pt"
            )
            
            # Load saved pruned parameters
            pruned_params = torch.load(self.cache_dir / "test_sae.pt")
            
            # Verify shapes
            self.assertEqual(pruned_params['W_dec'].shape, (2, 128))  # Only 2 features kept
            self.assertEqual(pruned_params['W_enc'].shape, (128, 2))  # Matching encoder shape
            self.assertEqual(pruned_params['b_enc'].shape, (2,))      # 2 biases
            self.assertEqual(pruned_params['threshold'].shape, (2,))   # 2 thresholds
            
            # Verify content - check if the saved weights match the original weights at indices 0 and 1
            self.assertTrue(torch.allclose(
                pruned_params['W_dec'], 
                torch.from_numpy(sae_params['W_dec'][[3, 7], :])
            ))
            self.assertTrue(torch.allclose(
                pruned_params['W_enc'], 
                torch.from_numpy(sae_params['W_enc'][:, [3, 7]])
            ))
            self.assertTrue(torch.allclose(
                pruned_params['b_enc'], 
                torch.from_numpy(sae_params['b_enc'][[3, 7]])
            ))
            self.assertTrue(torch.allclose(
                pruned_params['threshold'], 
                torch.from_numpy(sae_params['threshold'][[3, 7]])
            ))

            # Verify API was called correctly
            mock_get.assert_called_with(
                "https://www.neuronpedia.org/api/feature/features/test/3",
                headers={"X-Api-Key": "fake_key"}
            )

            save_pruned_sae(
                metadata_path=metadata_path,
                dump_dir=self.cache_dir,
                modified_refs=[9],
                savefile="single_concept_sae.pt",
                sae_params=sae_params,
                concept_ids=[1],
            )
            single = torch.load(
                self.cache_dir / "single_concept_sae.pt", weights_only=True
            )
            self.assertEqual(single["W_dec"].shape, (1, 128))
            self.assertTrue(torch.equal(
                single["W_dec"], torch.from_numpy(sae_params["W_dec"][[9]])
            ))

    def test_load(self):
        """Test model load with different intervention types"""
        # Create dummy SAE parameters
        dummy_params = {
            'W_dec': torch.randn(2, 128),  # 2 features, 128 dimensions
            'W_enc': torch.randn(128, 2),  # Matching encoder weights
            'b_dec': torch.randn(128),     # Decoder bias
            'b_enc': torch.randn(2),       # Encoder bias
            'threshold': torch.randn(2)     # Thresholds for each feature
        }
        
        # Save dummy parameters
        torch.save(dummy_params, self.cache_dir / "GemmaScopeSAE.pt")
        torch.save(
            torch.tensor([12.0, 15.0]),
            self.cache_dir / "GemmaScopeSAE_scale.pt",
        )
        
        # Create mock training args
        mock_training_args = MagicMock()
        mock_training_args.batch_size = 32
        mock_training_args.lr = 1e-4
        mock_training_args.weight_decay = 0.01
        
        # Create a mock transformer model with required config
        mock_config = MagicMock()
        mock_config.hidden_size = 128
        mock_model = MagicMock()
        mock_model.config = mock_config
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        
        # Test both steering and latent modes
        test_modes = [
            {"mode": "steering", "intervention_type": "addition"},  # Should use AdditionIntervention
            {"mode": "steering", "intervention_type": "clamping"},  # Should use DictionaryAdditionIntervention
            {"mode": "latent", "intervention_type": None},         # Should use JumpReLUSAECollectIntervention
        ]
        
        for test_config in test_modes:
            # Initialize model with all required parameters
            model = GemmaScopeSAE(
                model=mock_model,
                tokenizer=mock_tokenizer,
                layer=10,
                training_args=mock_training_args,
                device="cpu",
                seed=42,
                steering_layers=[10],
                dump_dir=self.cache_dir
            )
            
            # Load the saved parameters with specific mode
            model.load(
                dump_dir=self.cache_dir,
                mode=test_config["mode"],
                intervention_type=test_config["intervention_type"],
                # The checkpoint shape must override an evaluator default.
                low_rank_dimension=999,
            )
            
            # Verify the loaded parameters match based on intervention type
            if test_config["mode"] == "steering" and test_config["intervention_type"] == "addition":
                print("testing with AdditionIntervention")
                self.assertIsInstance(model.ax, AdditionIntervention)
                self.assertTrue(torch.allclose(model.ax.proj.weight.data, dummy_params['W_dec']))
            
            elif test_config["mode"] == "steering" and test_config["intervention_type"] == "clamping":
                print("testing with DictionaryAdditionIntervention")
                self.assertIsInstance(model.ax, DictionaryAdditionIntervention)
                self.assertTrue(torch.allclose(model.ax.W_dec, dummy_params['W_dec']))
                self.assertTrue(torch.allclose(model.ax.W_enc, dummy_params['W_enc']))
                self.assertTrue(torch.allclose(model.ax.b_dec, dummy_params['b_dec']))
                self.assertTrue(torch.allclose(model.ax.b_enc, dummy_params['b_enc']))
                self.assertTrue(torch.allclose(model.ax.threshold, dummy_params['threshold']))
            
            else:  # latent mode
                print("testing with JumpReLUSAECollectIntervention")
                self.assertIsInstance(model.ax, JumpReLUSAECollectIntervention)
                self.assertTrue(torch.allclose(model.ax.W_dec, dummy_params['W_dec']))
                self.assertTrue(torch.allclose(model.ax.W_enc, dummy_params['W_enc']))
                self.assertTrue(torch.allclose(model.ax.b_dec, dummy_params['b_dec']))
                self.assertTrue(torch.allclose(model.ax.b_enc, dummy_params['b_enc']))
                self.assertTrue(torch.allclose(model.ax.threshold, dummy_params['threshold']))

    def test_pre_compute_mean_activations(self):
        """Test pre_compute_mean_activations with mocked API responses"""
        # Create dummy parquet data
        import pandas as pd
        
        # Create test data directory
        test_data_dir = self.cache_dir / "data"
        test_data_dir.mkdir(exist_ok=True)
        
        # Create dummy latent data parquet file
        df = pd.DataFrame({
            "sae_link": [
                "https://www.neuronpedia.org/model1/sae1/0",
                "https://www.neuronpedia.org/model1/sae1/1"
            ]
        })
        df.to_parquet(self.cache_dir / "latent_data_test.parquet")
        
        # Mock API responses
        mock_responses = {
            "https://www.neuronpedia.org/api/feature/model1/sae1/0": {
                "activations": [{"maxValue": 75.5}]
            },
            "https://www.neuronpedia.org/api/feature/model1/sae1/1": {
                "activations": [{"maxValue": -25.0}]
            }
        }
        
        # Initialize model
        model = GemmaScopeSAE(
            model=MagicMock(),
            tokenizer=MagicMock(),
            layer=10,
            training_args=MagicMock(),
            device="cpu",
            seed=42,
            steering_layers=[10],
            dump_dir=self.cache_dir
        )
        
        with patch('requests.get') as mock_get, \
             patch.dict(os.environ, {'NP_API_KEY': 'fake_key'}):
            
            # Configure mock to return different responses based on URL
            mock_get.side_effect = lambda url, headers: MagicMock(
                json=lambda: mock_responses[url]
            )
            
            # Call the function
            max_activations = model.pre_compute_mean_activations(
                dump_dir=self.cache_dir,
                master_data_dir=test_data_dir
            )
            
            # Verify API calls
            expected_calls = [
                call('https://www.neuronpedia.org/api/feature/model1/sae1/0', 
                     headers={'X-Api-Key': 'fake_key'}),
                call('https://www.neuronpedia.org/api/feature/model1/sae1/1', 
                     headers={'X-Api-Key': 'fake_key'})
            ]
            mock_get.assert_has_calls(expected_calls, any_order=True)
            
            # Verify results
            expected_activations = {
                0: 75.5,
                1: 50.0  # Note: negative values are replaced with 50
            }
            self.assertEqual(max_activations, expected_activations)
            
            # Verify cache file was created
            cache_file = test_data_dir / "model1_sae1_max_activations.json"
            self.assertTrue(cache_file.exists())
            
            # Verify cached content
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            self.assertEqual(cached_data, {str(k): v for k, v in expected_activations.items()})
            
            # Test loading from cache
            # Clear the mock to verify no new API calls are made
            mock_get.reset_mock()
            
            # Call function again - should use cached values
            cached_activations = model.pre_compute_mean_activations(
                dump_dir=self.cache_dir,
                master_data_dir=test_data_dir
            )
            
            # Verify no new API calls were made
            mock_get.assert_not_called()
            
            # Verify cached values are returned correctly
            self.assertEqual(cached_activations, expected_activations)

    def tearDown(self):
        """Clean up after each test"""
        torch.cuda.empty_cache()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class"""
        # Remove cache directory and its contents
        shutil.rmtree(cls.cache_dir)
        torch.cuda.empty_cache()
