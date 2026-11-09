"""Fine-tune T5 to turn (subject, predicate, object) triplets into sentences, then evaluate it.

Examples:
    python main.py --max-samples 20000                    # REBEL subset, code defaults
    python main.py --preset paper --max-samples 20000     # hyper-parameters stated in main.tex
    python main.py --data examples/sample_triplets.csv --model t5-small --epochs 1 --no-bertscore
"""
import argparse
import json
import logging
from pathlib import Path

from data_preprocessing import DataPreprocessor
from evaluation import ModelEvaluator

PRESETS = {
    # What this code used originally.
    'code': {'model': 't5-small', 'epochs': 10, 'batch_size': 64, 'learning_rate': 5e-5},
    # What the paper (main.tex, "Implementation Details") describes.
    'paper': {'model': 't5-base', 'epochs': 3, 'batch_size': 8, 'learning_rate': 2e-5},
}

EXAMPLE_TRIPLETS = [
    ('Albert Einstein', 'developed', 'theory of relativity'),
    ('William Shakespeare', 'wrote', 'Hamlet'),
]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='code',
                        help='default hyper-parameters (individual flags override them)')
    parser.add_argument('--model', help='Hugging Face model id or local path')
    parser.add_argument('--epochs', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='stop after this many optimizer steps (for quick smoke runs)')
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--data', help='local .csv/.json/.jsonl with subject,predicate,object,text '
                                       '(default: stream REBEL from the Hugging Face Hub)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='use at most this many rows (REBEL has ~3M sentences)')
    parser.add_argument('--all-triplets', action='store_true',
                        help='REBEL: also use sentences with several triplets (first one is used)')
    parser.add_argument('--output-dir', default='./t5_triplet_model')
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--no-bertscore', action='store_true',
                        help='skip BERTScore (avoids downloading roberta-large)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None):
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args(argv)
    logger.info('Configuration: %s', vars(args))

    preprocessor = DataPreprocessor(max_samples=args.max_samples,
                                    single_triplet=not args.all_triplets, seed=args.seed)
    logger.info('Loading and preprocessing data...')
    loaded = (preprocessor.load_local_dataset(args.data) if args.data
              else preprocessor.load_rebel_dataset())
    if not loaded:
        logger.error('Failed to load dataset')
        return 1
    if not preprocessor.clean_data():
        logger.error('Failed to clean dataset')
        return 1
    train_data, val_data = preprocessor.get_processed_data()
    logger.info('Processed data: %d training samples, %d validation samples',
                len(train_data), len(val_data))

    from model import TripletToSentenceModel  # imports torch/transformers
    model = TripletToSentenceModel(args.model)
    train_dataset, val_dataset = model.prepare_dataset(train_data, val_data)

    logger.info('Starting model training...')
    model.train(train_dataset, val_dataset, output_dir=args.output_dir, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps, max_steps=args.max_steps, seed=args.seed)
    logger.info('Model training completed; saved to %s', args.output_dir)

    logger.info('Evaluating model...')
    evaluator = ModelEvaluator(use_bertscore=not args.no_bertscore)
    metrics, generated, references = evaluator.evaluate_model(model, val_data,
                                                              batch_size=args.eval_batch_size)

    logger.info('Evaluation results:')
    logger.info('BLEU: %.4f', metrics['bleu'])
    for name, value in metrics['rouge'].items():
        logger.info('%s F1: %.4f', name.upper(), value)
    logger.info('Token-overlap precision / recall / F1: %.4f / %.4f / %.4f',
                metrics['precision'], metrics['recall'], metrics['f1'])
    if metrics['bertscore']:
        bs = metrics['bertscore']
        logger.info('BERTScore precision / recall / F1: %.4f / %.4f / %.4f',
                    bs['precision'], bs['recall'], bs['f1'])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    with open(output_dir / 'predictions.jsonl', 'w') as fh:
        for row, gen in zip(val_data.to_dict('records'), generated):
            fh.write(json.dumps({**row, 'generated': gen}) + '\n')
    logger.info('Wrote metrics.json and predictions.jsonl to %s', output_dir)

    logger.info('Example generations:')
    for triplet, sentence in zip(EXAMPLE_TRIPLETS, model.generate_batch(EXAMPLE_TRIPLETS)):
        logger.info('%s -> %s', triplet, sentence)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
