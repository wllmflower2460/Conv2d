import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                   stride=1, dilation=dilation_size, 
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs,
                                                   kernel_size, stride=stride, 
                                                   padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs,
                                                   kernel_size, stride=stride, 
                                                   padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNVAE(nn.Module):
    def __init__(self, input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, 
                 sequence_length=100, num_activities=12):
        super(TCNVAE, self).__init__()
        
        # TCN Encoder
        self.tcn_encoder = TemporalConvNet(input_dim, hidden_dims)
        
        # VAE components
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dims[-1])
        self.tcn_decoder = TemporalConvNet(hidden_dims[-1], hidden_dims[::-1] + [input_dim])
        
        # Activity classifier (for pretraining)
        self.activity_classifier = nn.Linear(latent_dim, num_activities)
        
        # Domain adaptation
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 domains: PAMAP2, UCI-HAR, TartanIMU
        )
    
    def encode(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)
        h = self.tcn_encoder(x)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # Global average pooling
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, sequence_length):
        h = self.decoder_fc(z)
        h = h.unsqueeze(-1).expand(-1, -1, sequence_length)
        x_recon = self.tcn_decoder(h)
        return x_recon.transpose(1, 2)
    
    def forward(self, x, alpha=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(1))
        
        # Activity prediction
        activity_logits = self.activity_classifier(z)
        
        # Domain prediction (for adversarial training)
        domain_logits = self.domain_classifier(ReverseLayerF.apply(z, alpha))
        
        return x_recon, mu, logvar, activity_logits, domain_logits

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None