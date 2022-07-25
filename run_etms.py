import subprocess
import argparse


parser = argparse.ArgumentParser(description='The Embedded Topic Model')

# data and file related arguments
parser.add_argument('--lang_code', type=str, default="en", help="The language of the tweets")
parser.add_argument('--max_topics', type=int, default=100, help='maximal number of topics')
parser.add_argument('--min_topics', type=int, default=25, help="minimal number of topics")
parser.add_argument('--device', type=int, default=1, help="which gpu to use..")
args = parser.parse_args()


for x in range(args.min_topics, args.max_topics+5, 5):
    command = f"CUDA_VISIBLE_DEVICES={args.device} python -m models.ETModel.main --mode train --num_topics {x}  --lang_code {args.lang_code}"
    subprocess.run(command, shell=True, check=True, capture_output=True)


