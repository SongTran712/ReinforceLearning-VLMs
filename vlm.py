import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
import os

class PatchEmbeddings(nn.Module):
    def __init__(
        self, img_size: int = 96, patch_size: int = 16, hidden_dim: int = 512
    ) -> None:
        super().__init__()
        # Store the input image size, the patch size and hidden dimension
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Calculate the total number of patches
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # Create a convolution to extract patch embeddings
        # in_channels=3 asummes a 3-channel image (RGB)
        # outp_channels=hidden_dim sets the number of output channels to match the hidden dimension
        # kernel_size=patch_size and stride=patch_size ensuring each patch is embedded separately
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Extract patch embeddings from the input image
        # Output shape: (batch_size, hidden_dim, (self.img_size // self.patch_size), (self.img_size // self.patch_size))
        X = self.conv(X)

        # Flatten the spatial dimensions (height and width) of the patch embeddings
        # This step flattens the patch dimensions to a single dimension
        # Output shape: (batch_size, hidden_dim, self.num_patches)
        X = X.flatten(2)

        # Transpose the dimensions to obtain the shape (batch_size, num_patches, hidden_dim)
        # This step brings the num_patches dimension to the second position
        # Output shape: (batch_size, self.num_patches, hidden_dim)
        X = X.transpose(1, 2)

        return X
    

class Head(nn.Module):
    def __init__(
        self,
        n_embed: int,
        head_size: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()

        # Linear layer for Key projection
        self.key = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)

        # Linear layer for Query projection
        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)

        # Linear layer for Value projection
        self.value = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)

        # Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

        # Flag indicating wheter the head is used as a decoder
        self.is_decoder = is_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get batch size (B), sequence length (T), and embedding dimension (C) from the input tensor
        B, T, C = x.shape

        # Compute Key, Query, and Value projections
        k = self.key(x)  # Shape: (B, T, head_size)
        q = self.query(x)  # Shape: (B, T, head_size)
        v = self.value(x)  # SHape: (B, T, head_size)

        # Compute attention scores by taking the dot product of Query and Key
        # and scaling by the square root of the embedding dimension
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # Shape: (B, T, T)

        if self.is_decoder:
            # If this head is used in the decoder, apply causal mask to the attention scores
            # to prevent attention to future positions
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(mask=tril == 0, value=float("-inf"))

        # Apply softmax to the attention scores to obtain attention probabilities
        # Sum of probabilities for each row will be 1
        wei = F.softmax(input=wei, dim=-1)  # Shape: (B, T, T)

        # Apply Dropout to the attention probabilities for regularization
        wei = self.dropout(wei)

        # Perform a weighted aggregation of values using the attention probabilities
        out = wei @ v  # Shape: (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embed: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()

        # Ensure that the embedding dimension is divisible by the number of heads
        assert n_embed % num_heads == 0, "n_embed must be divisible by num_heads!"

        # Create a ModuleList of attention heads
        self.heads = nn.ModuleList(
            modules=[
                Head(
                    n_embed=n_embed,
                    head_size=n_embed // num_heads,
                    dropout=dropout,
                    is_decoder=is_decoder,
                )
                for _ in range(num_heads)
            ]
        )

        # Linear layer for projecting the concatenated head outputs
        self.proj = nn.Linear(in_features=n_embed, out_features=n_embed)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each attention head to the input tensor
        head_outputs = [
            h(x) for h in self.heads
        ]  # Shape: num_heads * (B, T, head_size)

        # Concatenate the outputs from all heads along the last dimension
        out = torch.cat(tensors=head_outputs, dim=-1)  # Shape: (B, T, m_embed)

        # Apply the projection layer to the concatenated outputs
        out = self.proj(out)  # Shape: (B, T, m_embed)

        # Apply Dropout to the projected outputs for regularization
        out = self.dropout(out)

        return out
    

class MLP(nn.Module):
    def __init__(
        self, n_embed: int, dropout: float = 0.1, is_decoder: bool = False
    ) -> None:
        super().__init__()

        # Define the layers of the MLP
        layers = [
            # First linear layer that expands the input dimension from n_embed to 4 * n_embed
            nn.Linear(in_features=n_embed, out_features=4 * n_embed),
            # Activation function: ReLU if is_decoder is True, else GELU
            nn.ReLU() if is_decoder else nn.GELU(),
            # Second linear layer that projects the intermediate dimension back to n_embed
            nn.Linear(in_features=4 * n_embed, out_features=n_embed),
            # Dropout layer for regularization
            nn.Dropout(p=dropout),
        ]

        # Create the MLP as a sequence of layers
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the MLP layers
        return self.net(x)
    

class Block(nn.Module):
    def __init__(
        self,
        n_embed: int,
        num_heads: int,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ) -> None:
        super().__init__()

        # Layer normalization for the input to the attention layer
        self.ln1 = nn.LayerNorm(normalized_shape=n_embed)

        # Multi-head attention module
        self.mhattn = MultiHeadAttention(
            n_embed=n_embed, num_heads=num_heads, dropout=dropout, is_decoder=is_decoder
        )

        # Layer normalization for the input to the FFN
        self.ln2 = nn.LayerNorm(normalized_shape=n_embed)

        # Feed-forward neural network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=n_embed, out_features=4 * n_embed),
            nn.GELU(),  # Activation function
            nn.Linear(
                in_features=4 * n_embed, out_features=n_embed
            ),  # Projection back to the original dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Saving the input for residual connection
        original_x = x

        # Apply layer normalization to the input
        x = self.ln1(x)

        # Apply multi-head attention
        mhattn_output = self.mhattn(x)

        # Add the residual connection (original input) to the attention output
        x = original_x + mhattn_output

        # Apply later normalization to the input to the FFN
        x = self.ln2(x)

        # Apply the FFN
        ffn_output = self.ffn(x)

        # Apply the residual connection (input to the FFN) to the FFN output
        x = x + ffn_output

        return x
    

class ViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_hiddens: int,
        num_heads: int,
        num_blocks: int,
        emb_dropout: float,
        block_dropout: float,
    ) -> None:
        super().__init__()

        # Patch embedding layer to convert the input image into patches
        self.patch_embedding = PatchEmbeddings(
            img_size=img_size, patch_size=patch_size, hidden_dim=num_hiddens
        )

        # Learnable classification token
        self.cls_token = nn.Parameter(data=torch.zeros(size=(1, 1, num_hiddens)))

        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2

        # Learnable position embedding
        self.pos_embedding = nn.Parameter(
            data=torch.randn(size=(1, num_patches + 1, num_hiddens))
        )

        # Dropout layer for the embeddings
        self.dropout = nn.Dropout(p=emb_dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embed=num_hiddens,
                    num_heads=num_heads,
                    dropout=block_dropout,
                    is_decoder=False,
                )
                for _ in range(num_blocks)
            ]
        )

        # Layer normalization for the final representation
        self.layer_norm = nn.LayerNorm(normalized_shape=num_hiddens)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Convert the input image into patch embeddings
        x = self.patch_embedding(X)  # Shape: (B, num_patches, num_hiddens)

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # Shape: (B, 1, num_hiddens)

        # Concatenate the classification token with the patch embeddings
        x = torch.cat(
            tensors=(cls_tokens, x), dim=1
        )  # Shape: (B, num_patches + 1, num_hiddens)

        # Add the position embedding to the patch embeddings
        x += self.pos_embedding  # Shape: (B, num_patches + 1, num_hiddens)

        # Apply dropout to the embeddings
        x = self.dropout(x)  # Shape: (B, num_patches + 1, num_hiddens)

        # Pass the embeddings through the transformer blocks
        for block in self.blocks:
            x = block(x)  # Shape: (B, num_patches + 1, num_hiddens)

        # Apply layer normalization to the `[CLS]` token's final representation
        x = self.layer_norm(x[:, 0])  # Shape: (B, num_hiddens)

        return x


class MultiModalProjector(nn.Module):
    def __init__(
        self,
        n_embed: int,
        img_embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Define the projection network
        self.net = nn.Sequential(
            # Linear layer to expand the image embedding dimension
            nn.Linear(in_features=img_embed_dim, out_features=4 * img_embed_dim),
            # GELU activation function
            nn.GELU(),
            # Linear layer to project the expanded image embeddings to the text embedding dimension
            nn.Linear(in_features=4 * img_embed_dim, out_features=n_embed),
            # Dropout layer for regularization
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the projection network
        x = self.net(x)  # Shape: (B, img_embed_dim) --> (B, n_embed)
        return x


class DecoderLanguageModel(nn.Module):
    def __init__(
        self,
        n_embed: int,
        img_embed_dim: int,
        vocab_size: int,
        num_heads: int,
        n_layer: int,
        use_images: bool = False,
    ) -> None:
        super().__init__()

        self.use_images = use_images

        # Token embedding table
        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embed
        )

        # Position embedding table
        self.position_embedding_table = nn.Embedding(
            num_embeddings=1000, embedding_dim=n_embed
        )

        if use_images:
            # Image projection layer to align image embeddings with text embeddings
            self.image_projection = MultiModalProjector(
                n_embed=n_embed, img_embed_dim=img_embed_dim
            )

        # Stack of transformer decoder blocks
        self.blocks = nn.Sequential(
            *[
                Block(n_embed=n_embed, num_heads=num_heads, is_decoder=True)
                for _ in range(n_layer)
            ]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(normalized_shape=n_embed)

        # Language modeling head
        self.lm_head = nn.Linear(in_features=n_embed, out_features=vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        img_embeds: torch.Tensor = None,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:
        # Get token embeddings from the input indices
        tok_emb = self.token_embedding_table(idx)

        if self.use_images:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(img_embeds).unsqueeze(1)
            tok_emb = torch.cat([img_emb, tok_emb], dim=1)

        # Get position embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(tok_emb.size(1), device=idx.device)
        )

        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb

        # Pass through the transformer decoder blocks
        x = self.blocks(x)

        # Apply final layer normalization
        x = self.ln_f(x)

        # Get the logits from the language modeling head
        logits = self.lm_head(x)

        if targets is not None:
            if self.use_images and img_embeds is not None:
                # Prepare targets by concatenating a dummy target for the image embedding
                batch_size = idx.size(0)
                targets = torch.cat(
                    [
                        torch.full(
                            (batch_size, 1), -100, dtype=torch.long, device=idx.device
                        ),
                        targets,
                    ],
                    dim=1,
                )

            # Compute the cross-entropy loss
            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),
                target=targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        return logits

    def generate(
        self, idx: torch.Tensor, img_embeds: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        # Get the batch size and sequence length
        B, T = idx.shape

        # Initialize the generated sequence with the input indices
        generated = idx
        tok_emb = self.token_embedding_table(idx)

        if self.use_images and img_embeds is not None:
            # Project and concatenate image embeddings with token embeddings
            img_emb = self.image_projection(img_embeds).unsqueeze(1)
            current_output = torch.cat([img_emb, tok_emb], dim=1)
        else:
            current_output = tok_emb

        # Generate new tokens iteratevely
        for i in range(max_new_tokens):
            # Get the current sequence length
            T_current = current_output.shape[1]

            # Get position embeddings for the current sequence length
            current_pos_emb = self.position_embedding_table(
                torch.arange(T_current, device=idx.device)
            ).unsqueeze(0)

            # Add position embeddings to the current output
            current_output += current_pos_emb

            # Pass through the transformer decoder blocks
            for block in self.blocks:
                current_output = block(current_output)

            # Get the logits for the last token
            logits = self.lm_head(current_output[:, -1, :])

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token based on the probability
            idx_next = torch.multinomial(input=probs, num_samples=1)

            # Concatenate the generated token to the generated sequence
            generated = torch.cat([generated, idx_next], dim=1)

            # Get the embeddings for the generated token
            idx_next_emb = self.token_embedding_table(idx_next)

            # Concatenate the generated token embeddings to the current output
            current_output = torch.cat([current_output, idx_next_emb], dim=1)

        return generated
    

class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        n_embed: int,
        img_embed_dim: int,
        vocab_size: int,
        n_layer: int,
        img_size: int,
        patch_size: int,
        num_heads: int,
        num_blocks: int,
        emb_dropout: float,
        block_dropout: float,
    ) -> None:
        super().__init__()

        # Set num_hiddens equal to img_embed_dim
        num_hiddens = img_embed_dim

        # Assert that num_hiddens is divisible by num_heads
        assert num_hiddens % num_heads == 0, ValueError(
            "num_hiddens must be divisible by num_heads!"
        )

        # Initialize the Vision Transformer (ViT) encoder
        self.vision_encoder = ViT(
            img_size=img_size,
            patch_size=patch_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            num_blocks=num_blocks,
            emb_dropout=emb_dropout,
            block_dropout=block_dropout,
        )

        # Initialize the Language Model Decoder (DecoderLanguageModel)
        self.decoder = DecoderLanguageModel(
            n_embed=n_embed,
            img_embed_dim=img_embed_dim,
            vocab_size=vocab_size,
            num_heads=num_heads,
            n_layer=n_layer,
            use_images=True,
        )

    def _check_image_embeddings(self, image_embeds: torch.Tensor) -> None:
        """Chek if image embeddings are valid."""
        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError(
                "Something is wrong with the ViT model. It's returning an empty tensor or the embedding dimension is empty."
            )

    def forward(
        self, img_array: torch.Tensor, idx: torch.Tensor, targets: torch.Tensor = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Get the image embeddings from the Vision Encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if image embeddings are valid
        self._check_image_embeddings(image_embeds)

        if targets is not None:
            # If targets are provided, compute the logits and loss
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            # If targets are not provided, compute only the logits
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(
        self, img_array: torch.Tensor, idx: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        # Get the image embeddings from the Vision Encoder
        image_embeds = self.vision_encoder(img_array)

        # Check if image embeddings are valid
        self._check_image_embeddings(image_embeds)

        # Generate new tokens using the Language Model Decoder
        generated_tokens = self.decoder.generate(
            idx=idx, img_embeds=image_embeds, max_new_tokens=max_new_tokens
        )
        return generated_tokens

# Preprocessing function
def preprocess(example):
    tokenizer = spm.SentencePieceProcessor(model_file='spm.model')
    max_len = 256
    
    img_data = example["images"][0]

    # Đảm bảo ảnh là PIL.Image
    if isinstance(img_data, Image.Image):
        img = img_data.convert("RGB")
    elif isinstance(img_data, np.ndarray):
        img = Image.fromarray(img_data).convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {type(img_data)}")

    # Transform để ra Tensor
    image_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    image = image_transform(img)  # Tensor (3, 224, 224)

    # Tokenize prompt và target
    prompt = example["texts"][0]["user"]
    target = example["texts"][0].get("assistant", "")

    # full_input = prompt + "\n" + target if target else prompt

    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0
    tokens = tokenizer.encode(prompt)
    tokens = tokens[:max_len]
    tokens += [pad_id] * (max_len - len(tokens))
    input_ids = torch.tensor(tokens, dtype=torch.long)

    # Tokenize target
    if target:
        target_tokens = tokenizer.encode(target)
        target_tokens = target_tokens[:max_len]
        target_tokens += [pad_id] * (max_len - len(target_tokens))
        target_ids = torch.tensor(target_tokens, dtype=torch.long)
    else:
        target_ids = torch.full_like(input_ids, fill_value=pad_id)

    return {
        "image": image,
        "input_ids": input_ids,
        "target_ids": target_ids
    }

def collate_fn(batch):
    # Đảm bảo chuyển về tensor đúng shape
    imgs = torch.stack([torch.tensor(item['image']) if not isinstance(item['image'], torch.Tensor) else item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    return imgs, input_ids, target_ids

def train():
    n_embed, num_hiddens, num_heads, n_layer = 128, 512, 8, 8
    image_embed_dim = num_hiddens
    img_size = 96
    patch_size = 16
    num_blocks = 2
    n_layer, block_size, num_hiddens = 8, 32, 512

    tokenizer = spm.SentencePieceProcessor(model_file='spm.model')
    max_len = 256

    # Initialize the model
    vlm = VisionLanguageModel(
        n_embed=n_embed,
        img_embed_dim=image_embed_dim,
        vocab_size=tokenizer.vocab_size(),
        n_layer=n_layer,
        img_size=img_size,
        patch_size=patch_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        emb_dropout=0.1,
        block_dropout=0.1,
    )
    device = torch.device('cpu')
    vlm.to(device)

    # optimizer = torch.optim.AdamW(vlm.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(vlm.parameters(), lr=0.001, momentum=0.9)

    # Load dataset
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "ai2d", split="train")
    save_path = "/dataset/ai2d"

    # Save the dataset to disk
    dataset.save_to_disk(save_path)
    # Lọc những entry có ảnh và text hợp lệ
    dataset = dataset.filter(lambda x: x["images"] and x["texts"] and "user" in x["texts"][0])

    # # Lấy 100 sample đầu tiên, lấy full dataset thì bỏ qua dòng này
    # dataset = dataset.select(range(100))

    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn) 

    best_loss = float("inf")
    # start_epoch = 0
    max_epochs = 100
    checkpoint_path = '/checkpoints'
    best_ckpt_path = os.path.join(checkpoint_path, "vlm_best.pt")

    if os.path.exists(best_ckpt_path):
        print("Found best checkpoint. Loading and continuing training...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        vlm.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("No checkpoint found. Starting training from scratch.")

    vlm.train()
    for epoch in range(max_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        total_loss = 0

        for imgs, input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            imgs = imgs.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            _, loss = vlm(imgs, input_ids, targets=target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': vlm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, best_ckpt_path)
            print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

if __name__ == '__main__':
    train()


