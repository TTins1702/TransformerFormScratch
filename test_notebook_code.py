import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

# Copy exactly the updated PyTorch model classes from the notebook to test them

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        # Apply Attention Dropout
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x, attn_weights

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        norm_x = self.norm1(x)
        self_attn_out, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        norm_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(norm_x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_out)
        
        norm_x = self.norm3(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x, self_attn_weights, cross_attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1, embedding=None):
        super(Encoder, self).__init__()
        self.d_model = d_model
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Scale embedding output by sqrt(d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        attn_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_list.append(attn_weights)
        return self.norm(x), attn_list

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1, embedding=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # Scale embedding output by sqrt(d_model)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        self_attn_list = []
        cross_attn_list = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, src_mask, tgt_mask)
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)
        return self.norm(x), self_attn_list, cross_attn_list

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=3, num_heads=8, d_ff=512, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        
        if src_vocab_size == tgt_vocab_size:
            self.shared_embedding = nn.Embedding(src_vocab_size, d_model)
            self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, embedding=self.shared_embedding)
            self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout, embedding=self.shared_embedding)
            self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
            self.generator.weight = self.shared_embedding.weight
        else:
            self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
            self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
            self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
            self.generator.weight = self.decoder.embedding.weight
            
        # Apply custom weights initialization first
        self.apply(self._init_weights)
        
        # Shared weights between Decoder embedding and pre-softmax linear transformation (tied weights)
        if src_vocab_size == tgt_vocab_size:
            self.generator.weight = self.shared_embedding.weight
        else:
            self.generator.weight = self.decoder.embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
        # Apply scaled initialization to residual projection layers
        if isinstance(module, EncoderLayer) or isinstance(module, DecoderLayer):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'W_o'):
                nn.init.normal_(module.self_attn.W_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            if hasattr(module, 'cross_attn') and hasattr(module.cross_attn, 'W_o'):
                nn.init.normal_(module.cross_attn.W_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            if hasattr(module, 'feed_forward') and hasattr(module.feed_forward, 'w_2'):
                nn.init.normal_(module.feed_forward.w_2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output, enc_attn = self.encoder(src, src_mask)
        dec_output, dec_self_attn, dec_cross_attn = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.generator(dec_output)
        return logits, enc_attn, dec_self_attn, dec_cross_attn

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        self.lr = 0.0

    def step(self):
        self.step_num += 1
        self.lr = self.factor * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

# Mask functions using .to(src.device) and .to(tgt.device)
def make_src_mask(src, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(src.device)

def make_tgt_mask(tgt, pad_idx=0):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    no_peak_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    no_peak_mask = no_peak_mask.unsqueeze(0).unsqueeze(1)
    tgt_mask = tgt_pad_mask & no_peak_mask
    return tgt_mask.to(tgt.device)

# --- Test script execution ---
def run_tests():
    print("Running PyTorch model shape and device compatibility tests...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test device: {device}")
    
    # 1. Test Single-GPU / CPU forward pass
    src_vocab_size = 100
    tgt_vocab_size = 100 # Share vocab to test shared embedding weight constraint
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=64, num_layers=2, num_heads=4, d_ff=128)
    model = model.to(device)
    
    src = torch.randint(1, src_vocab_size, (4, 12)).to(device) # batch_size=4, seq_len=12
    tgt = torch.randint(1, tgt_vocab_size, (4, 10)).to(device) # batch_size=4, seq_len=10
    
    tgt_input = tgt[:, :-1]
    tgt_target = tgt[:, 1:]
    
    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt_input)
    
    print("Testing forward pass...")
    logits, enc_attn, dec_self_attn, dec_cross_attn = model(src, tgt_input, src_mask, tgt_mask)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (4, 9, tgt_vocab_size), f"Expected logits shape (4, 9, {tgt_vocab_size}), got {logits.shape}"
    
    # Test shared embedding weight constraint
    assert model.generator.weight is model.decoder.embedding.weight, "Embedding weight should be shared with generator linear weight!"
    assert model.encoder.embedding.weight is model.decoder.embedding.weight, "Encoder embedding weight should be shared with Decoder embedding weight!"
    print("Shared embedding weight check passed.")
    
    print("Testing backward pass...")
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_target.contiguous().view(-1))
    loss.backward()
    print("Backward pass completed successfully.")
    
    # 2. Test Multi-GPU (nn.DataParallel Mocking/Testing)
    # Even if 1 GPU or CPU, we can wrap in nn.DataParallel and check device consistency.
    print("Testing nn.DataParallel wrapping...")
    dp_model = nn.DataParallel(model)
    dp_logits, _, _, _ = dp_model(src, tgt_input, src_mask, tgt_mask)
    print(f"DataParallel Logits shape: {dp_logits.shape}")
    
    # 3. Test device aware masking specifically
    dummy_device = torch.device("cpu")
    if torch.cuda.is_available():
        dummy_device = torch.device("cuda:0")
    
    src_dummy = torch.randint(1, src_vocab_size, (2, 8)).to(dummy_device)
    mask_dummy = make_src_mask(src_dummy)
    assert mask_dummy.device == dummy_device, f"Mask device {mask_dummy.device} should be {dummy_device}"
    print("Device-aware masking check passed.")
    
    # 4. Test NoamScheduler learning rate calculations
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    scheduler = NoamScheduler(optimizer, d_model=64, warmup_steps=10)
    lrs = []
    for _ in range(20):
        lrs.append(scheduler.step())
    assert lrs[9] > lrs[0], "Learning rate should increase during warmup"
    assert lrs[19] < lrs[9], "Learning rate should decay after warmup"
    print("Noam Learning Rate Scheduler check passed.")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_tests()
