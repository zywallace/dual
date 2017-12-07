from dual.model import Trainer
from onmt import Statistics
MAX_ITER = 10


def report_func(epoch, batch, num_batches,
                start_time, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % 10000 == -1 % 100000:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = Statistics()

    return report_stats


def main():
    trainer = Trainer(agent_a, agent_b)

    for epoch in range(MAX_ITER):
        print('')

        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Update the learning rate
        trainer.epoch_step(valid_stats.ppl(), epoch)

        # 4. Drop a checkpoint if needed.
        trainer.drop_checkpoint(opt, epoch, fields, valid_stats)


if __name__ == "__main__":
    main()