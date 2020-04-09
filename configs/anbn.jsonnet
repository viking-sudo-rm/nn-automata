local DIM = std.parseInt(std.extVar("DIM"));


{
  "dataset_reader": {
    "type": "anbn",
  },

  "train_data_path": "1:1000",
  "validation_data_path": "1000:1100",
  
  "model": {
    "type": "djanky_lm",
    "rnn_type": "srn",
    "rnn_dim": DIM,
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 16,
    },
  },

  "trainer": {
    "optimizer": "adam",
    "num_epochs": 100,
    "cuda_device": 0,
    "validation_metric": "-loss",
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    }
  }
}