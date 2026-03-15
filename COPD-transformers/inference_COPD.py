# ············· QUERALT MARTÍN-SALADICH, 2024 ··············
# ···················· BWH + UPF + VHIR ····················
# ··························································

from Utils.inference.predictions_COPD import Predictor_COPD
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="main path")
parser.add_argument("--cuda", help="cuda node id",default=1)
parser.add_argument("--model", help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--output", help="1 for emphysema, 2 for trajectories, 3 for COPD",default=3)
parser.add_argument("--project_name", help="prefix/suffix to the name of the files, e.g. COPDGene",default="COPDGene")

args = parser.parse_args()

main_path = args.path
cuda_id = args.cuda
model = args.model
project_name = args.project_name
output = args.output

for batch_size in [20]:
    predicting = Predictor_COPD(main_path=main_path,
                    model_id=int(model),
                    num_slices=batch_size,
                    cuda_id=int(cuda_id),
                    output_id=int(output),
                    project_name=project_name)
    
print('Finished')