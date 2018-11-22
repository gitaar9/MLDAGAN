import data as dataset
from experiment_builder import ExperimentBuilder
from utils.parser_util import get_args


# Run this with:
# python train_folio_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title folio22112018 --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5

batch_size, num_gpus, args = get_args()
#set the data provider to use for the experiment
data = dataset.FolioDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                 num_of_gpus=num_gpus, gen_batches=10)
#init experiment
experiment = ExperimentBuilder(args, data=data)
#run experiment
experiment.run_experiment()
