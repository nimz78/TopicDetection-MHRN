import torch
import torch.nn as nn
import torchvision.models as models


class MHRN(nn.Module):

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=1, num_topics=2):
        super(MHRN, self).__init__()

        # **Text Encoder: RNN (GRU)**
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # **Image Encoder: ResNet18 (Faster than ResNet50)**
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Smaller model
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.image_fc = nn.Linear(512, hidden_dim * 2)  # Adjust dimensions

        # **Fusion & Classification**
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim * 2, num_topics)  # Ensure dimensions match
        self.dropout = nn.Dropout(0.3)  # Reduce overfitting

    def forward(self, text, text_lengths, images):
        # **Text Processing**
        embedded = self.embedding(text)  # Shape: (batch_size, max_length, embed_dim)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)

        # If bidirectional, concatenate the last hidden states
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # Shape: (batch_size, hidden_dim * 2)
        else:
            hidden = hidden[-1, :, :]  # Shape: (batch_size, hidden_dim)

        print(f"RNN Output Shape: {hidden.shape}")  # Debugging

        # **Image Processing**
        image_features = self.cnn(images).squeeze()  # Shape: (batch_size, 2048)
        image_features = self.image_fc(image_features)  # Reduce to (batch_size, hidden_dim)

        print(f"Image Features Shape: {image_features.shape}")  # Debugging

        # **Fusion**
        fused = torch.cat((hidden, image_features), dim=1)  # Shape should match
        print(f"Fused Features Shape: {fused.shape}")  # Debugging
        output = self.fc(self.dropout(fused))

        return output
