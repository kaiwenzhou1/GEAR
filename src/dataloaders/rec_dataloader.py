

from .base import AbstractDataloader

import torch
import numpy as np
import torch.utils.data as data_utils

class RecDataloader(AbstractDataloader):
    def __init__(
            self,
            dataset,
            seg_len,
            mask_prob,
            num_items,
            num_users,
            num_workers,
            val_negative_sampler_code,
            val_negative_sample_size,
            train_batch_size,
            val_batch_size,
            predict_only_target=False,
        ):
        super().__init__(dataset,
            val_negative_sampler_code,
            val_negative_sample_size)
        self.target_code = self.bmap.get('buy') if self.bmap.get('buy') else self.bmap.get('pos')
        self.seg_len = seg_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.num_users = num_users
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.predict_only_target = predict_only_target
    
    def get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, num_workers=self.num_workers)
        return dataloader

    def _get_train_dataset(self):
        return RecTrainDataset(
            self.train, self.train_b, self.train_t,
            self.seg_len, self.mask_prob, 
            self.num_items, self.target_code,
            self.predict_only_target
        )

    def get_val_loader(self):
        dataset = self._get_eval_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.val_batch_size,
                                           shuffle=False, num_workers=self.num_workers)
        return dataloader

    def _get_eval_dataset(self):
        return RecEvalDataset(
            self.train, self.train_b, 
            self.val, self.val_b, 
            self.train_t, self.val_t, 
            self.val_num, self.seg_len,
            self.num_items, self.target_code,
            self.val_negative_samples
        )

class RecTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2b, u2t, max_len, mask_prob, num_items, target_code, predict_only_target):
        self.u2seq = u2seq
        self.u2b = u2b
        self.u2t = u2t
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.num_items = num_items
        self.target_code = target_code
        self.predict_only_target = predict_only_target

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        b_seq = self.u2b[user]
        t_seq = self.u2t[user] 

        tokens = seq.copy()
        behaviors = b_seq.copy()
        labels = [0] * len(seq) 

        if len(tokens) > self.max_len and np.random.rand() < 0.2:
            start_idx = np.random.randint(0, len(tokens) - self.max_len + 1)
        else:
            start_idx = max(0, len(tokens) - self.max_len)
        
        end_idx = start_idx + self.max_len

        tokens = tokens[start_idx:end_idx]
        behaviors = behaviors[start_idx:end_idx]
        time_gaps = t_seq[start_idx:end_idx]

        padding_len = self.max_len - len(tokens)
        tokens = [0] * padding_len + tokens
        behaviors = [0] * padding_len + behaviors
        time_gaps = [0] * padding_len + time_gaps

        cum_sum = np.cumsum([0] + time_gaps).tolist()
        bias_matrix = torch.tensor(
            [[cum_sum[i+1] - cum_sum[j+1] if j < i else 0 
              for j in range(self.max_len)] 
             for i in range(self.max_len)],
            dtype=torch.float
        )

        return {
            'user_id': torch.LongTensor([user]),
            'input_ids': torch.LongTensor(tokens),
            'labels': torch.LongTensor(labels[-self.max_len:] + [0]*padding_len), 
            'behaviors': torch.LongTensor(behaviors),
            'time_bias': bias_matrix
        }


class RecEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2b, u2answer, u2ab, u2t, u2at, val_num, max_len, num_items, target_code, negative_samples):
        self.u2seq = u2seq
        self.u2b = u2b
        self.u2answer = u2answer
        self.users = sorted(self.u2answer.keys())
        self.u2ab = u2ab
        self.u2t = u2t
        self.u2at = u2at
        self.val_num = val_num
        self.max_len = max_len
        self.negative_samples = negative_samples
        self.num_items = num_items
        self.target_code = target_code

    def __len__(self):
        return self.val_num

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.num_items + 1]
        seq = seq[-self.max_len:]
        seq_b = self.u2b[user] + self.u2ab[user]
        seq_b = seq_b[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        padding_len = self.max_len - len(seq_b)
        seq_b = [0] * padding_len + seq_b

        seq_t = self.u2t[user] + self.u2at[user] 
        seq_t = seq_t[-self.max_len:]
        padding_len = self.max_len - len(seq_t)
        time_gaps = [0] * padding_len + seq_t
        
        cum_sum = [0]
        for t in time_gaps:
            cum_sum.append(cum_sum[-1] + t)
        
        seq_len = len(time_gaps)
        bias_matrix = torch.zeros((seq_len, seq_len), dtype=torch.float)
        for i in range(seq_len):
            for j in range(i):
                bias_matrix[i, j] = cum_sum[i+1] - cum_sum[j+1]

        user = [user]
        
        return {
            'user_id':torch.LongTensor(user),
            'input_ids':torch.LongTensor(seq),
            'candidates':torch.LongTensor(candidates), 
            'labels':torch.LongTensor(labels),
            'behaviors': torch.LongTensor(seq_b),
            'time_bias': bias_matrix,
        }