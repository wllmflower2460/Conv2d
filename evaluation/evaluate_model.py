import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
sys.path.append('/home/wllmflower/tcn-vae-training')

from models.tcn_vae import TCNVAE
from preprocessing.unified_pipeline import MultiDatasetHAR

def evaluate_trained_model():
    """Evaluate the trained TCN-VAE model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processor
    with open('/home/wllmflower/tcn-vae-training/models/processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    
    # Load validation data
    X_train, y_train, domains_train, X_val, y_val, domains_val = processor.preprocess_all()
    
    # Load best model
    num_activities = len(np.unique(y_train))
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, num_activities=num_activities)
    model.load_state_dict(torch.load('/home/wllmflower/tcn-vae-training/models/best_tcn_vae.pth'))
    model.to(device)
    model.eval()
    
    # Prepare validation data
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
        torch.LongTensor(domains_val)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Generate comprehensive evaluation
    predictions = []
    true_labels = []
    latent_features = []
    
    print("Evaluating model on validation set...")
    
    with torch.no_grad():
        for data, labels, domains in val_loader:
            data = data.to(device).float()
            labels = labels.to(device).long()
            
            mu, _ = model.encode(data)
            activity_logits = model.activity_classifier(mu)
            
            predictions.extend(activity_logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            latent_features.extend(mu.cpu().numpy())
    
    # Classification report
    report = classification_report(true_labels, predictions)
    print("Activity Classification Report:")
    print(report)
    
    # Accuracy calculation
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/home/wllmflower/tcn-vae-training/evaluation/confusion_matrix.png')
    plt.close()
    
    # t-SNE visualization of latent space (sample subset for speed)
    if len(latent_features) > 1000:
        sample_idx = np.random.choice(len(latent_features), 1000, replace=False)
        latent_subset = np.array(latent_features)[sample_idx]
        labels_subset = np.array(true_labels)[sample_idx]
    else:
        latent_subset = np.array(latent_features)
        labels_subset = np.array(true_labels)
    
    print("Computing t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_subset)-1))
    latent_2d = tsne.fit_transform(latent_subset)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=labels_subset, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Learned Latent Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('/home/wllmflower/tcn-vae-training/evaluation/tsne_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, report

def export_for_edgeinfer():
    """Export model for EdgeInfer integration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processor for configuration
    with open('/home/wllmflower/tcn-vae-training/models/processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    
    # Load trained model
    X_train, y_train, _, _, _, _ = processor.preprocess_all()
    num_activities = len(np.unique(y_train))
    
    model = TCNVAE(input_dim=9, hidden_dims=[64, 128, 256], latent_dim=64, num_activities=num_activities)
    model.load_state_dict(torch.load('/home/wllmflower/tcn-vae-training/models/best_tcn_vae.pth'))
    model.to(device)
    
    # Export encoder for inference
    torch.save(model.tcn_encoder.state_dict(), '/home/wllmflower/tcn-vae-training/export/tcn_encoder_for_edgeinfer.pth')
    
    # Export full model for motif extraction
    torch.save(model.state_dict(), '/home/wllmflower/tcn-vae-training/export/full_tcn_vae_for_edgeinfer.pth')
    
    # Create model configuration file
    config = {
        'input_dim': 9,
        'hidden_dims': [64, 128, 256],
        'latent_dim': 64,
        'sequence_length': 100,
        'window_overlap': 0.5,
        'num_activities': num_activities,
        'activity_mapping': processor.label_mappings
    }
    
    import json
    with open('/home/wllmflower/tcn-vae-training/export/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Model exported successfully for EdgeInfer integration!")
    print("Exported files:")
    print("- tcn_encoder_for_edgeinfer.pth")
    print("- full_tcn_vae_for_edgeinfer.pth")
    print("- model_config.json")
    
    return config

if __name__ == "__main__":
    print("=== TCN-VAE Model Evaluation ===")
    
    try:
        accuracy, report = evaluate_trained_model()
        
        print(f"\n=== Final Results ===")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        if accuracy > 0.85:
            print("🎯 SUCCESS: Model achieved >85% accuracy target!")
        else:
            print(f"📊 Model achieved {accuracy:.4f} accuracy (target: >85%)")
        
        print("\n=== Exporting for EdgeInfer ===")
        config = export_for_edgeinfer()
        
        print("\n=== Evaluation Complete ===")
        print("Check the following files:")
        print("- evaluation/confusion_matrix.png")
        print("- evaluation/tsne_features.png")
        print("- export/ directory for EdgeInfer integration files")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()