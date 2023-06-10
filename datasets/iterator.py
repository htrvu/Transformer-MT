from torchtext.legacy import data

class MyIterator(data.Iterator):
    def create_batches(self):
        """
        Overrides create_batches method in torchtext.data.Iterator

        Yields:
            torchtext.data.Batch: batch of data
        """
        if self.train:

            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,
                        self.batch_size_fn,
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = list(pool(self.data(), self.random_shuffler))
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))