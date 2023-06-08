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


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
