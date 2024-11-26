from pathlib import Path
import torch
import numpy as np
import pandas as pd
import anndata
import yaml
from torch.utils.data import DataLoader, TensorDataset
from absl import app, flags
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import scipy.sparse

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', '', 'Path to saved model.pt file')
flags.DEFINE_string('config_path', '', 'Path to config.yaml file')
flags.DEFINE_string('outdir', '', 'Path to output directory')
flags.DEFINE_string(
    'human_adata',
    '/home/dor3/palmer_scratch/C2S_Files_Daphne/Cross_Species_Datasets/mouse_human_pancreas_tissue_Baron_et_al/processed_data/human_pancreas_preprocessed_log10_homolog_intersected_adata.h5ad',
    'Path to human AnnData file'
)
flags.DEFINE_string(
    'mouse_adata',
    '/home/dor3/palmer_scratch/C2S_Files_Daphne/Cross_Species_Datasets/mouse_human_pancreas_tissue_Baron_et_al/processed_data/mouse_pancreas_preprocessed_log10_homolog_intersected_adata.h5ad',
    'Path to mouse AnnData file'
)
flags.DEFINE_integer('n_reps', 10, 'Number of evaluation repetitions')

def load_config(config_path):
    """Load and parse config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_and_prepare_data(human_path, mouse_path, config):
    """Load and prepare the human and mouse data according to config"""
    # Load AnnData objects
    human_adata = anndata.read_h5ad(human_path)
    mouse_adata = anndata.read_h5ad(mouse_path)
    
    # Verify that genes match between datasets
    # assert set(human_adata.var_names) == set(mouse_adata.var_names), "Gene sets don't match"
    
    # Convert to torch tensors
    human_data = torch.FloatTensor(human_adata.X.toarray() if scipy.sparse.issparse(human_adata.X) else human_adata.X)
    mouse_data = torch.FloatTensor(mouse_adata.X.toarray() if scipy.sparse.issparse(mouse_adata.X) else mouse_adata.X)
    
    # Use config parameters for data splitting
    np.random.seed(config['datasplit']['random_state'])
    test_size = config['datasplit']['test_size']
    batch_size = config['dataloader']['batch_size']
    
    # Split data using config parameters
    human_indices = np.random.permutation(len(human_data))
    human_test_idx = human_indices[:test_size]
    human_test = human_data[human_test_idx]
    
    mouse_indices = np.random.permutation(len(mouse_data))
    mouse_test_idx = mouse_indices[:test_size]
    mouse_test = mouse_data[mouse_test_idx]
    
    # Create dataloaders using config batch size
    human_test_dataset = TensorDataset(human_test)
    mouse_test_dataset = TensorDataset(mouse_test)
    
    human_test_loader = DataLoader(
        human_test_dataset, 
        batch_size=batch_size, 
        shuffle=config['dataloader']['shuffle']
    )
    mouse_test_loader = DataLoader(
        mouse_test_dataset, 
        batch_size=batch_size, 
        shuffle=config['dataloader']['shuffle']
    )
    
    data_info = {
        'gene_names': human_adata.var_names,
        'human_cells': len(human_data),
        'mouse_cells': len(mouse_data),
        'n_genes': len(human_adata.var_names),
        'latent_dim': config['model']['latent_dim']
    }
    
    return human_test_loader, mouse_test_loader, data_info

def load_model(model_path, config, device):
    """Load model from checkpoint dictionary"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # If the model is saved as a dictionary with 'state_dict' key
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            # Print available keys to help debug
            print("Available keys in checkpoint:", checkpoint.keys())
            raise ValueError("Could not find model state in checkpoint")
            
        # Initialize a new model with config parameters
        model = CellOT(
            input_dim=config['data']['input_dim'],
            latent_dim=config['model']['latent_dim'],
            hidden_units=config['model']['g']['hidden_units'],
            fnorm_penalty=config['model']['g']['fnorm_penalty'],
            softplus_W_kernels=config['model']['softplus_W_kernels']
        )
        
        # Load the saved state
        model.load_state_dict(model_state)
    else:
        model = checkpoint  # In case the model was saved directly
        
    return model.to(device)


def compute_mmd_distance(x, y, gamma=1.0):
    """Compute MMD distance between two datasets"""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    K = torch.exp(- gamma * (rx.t() + rx - 2*xx))
    L = torch.exp(- gamma * (ry.t() + ry - 2*yy))
    P = torch.exp(- gamma * (rx.t() + ry - 2*xy))
    
    beta = (1./(x.size(0) * x.size(0)))
    gamma = (1./(y.size(0) * y.size(0)))
    
    return beta * (K.sum()) + gamma * (L.sum()) - 2.0 * beta * P.sum()

def compute_knn_enrichment(predicted, actual, k=100):
    """Compute kNN enrichment score"""
    nbrs = NearestNeighbors(n_neighbors=min(k, len(actual)), algorithm='ball_tree').fit(actual)
    distances, indices = nbrs.kneighbors(predicted)
    
    enrichment_scores = []
    for i in range(min(k, len(actual))):
        neighbors = indices[:, :(i+1)]
        score = np.mean([len(set(n)) / (i+1) for n in neighbors])
        enrichment_scores.append(score)
        
    return np.array(enrichment_scores)

def evaluate_model(model, human_loader, mouse_loader, data_info, device):
    """Evaluate model performance"""
    model.eval()
    metrics = {}
    
    # Collect predictions and actual values
    human_embeddings = []
    mouse_embeddings = []
    predicted_mouse = []
    actual_mouse_data = []
    
    with torch.no_grad():
        for human_batch in human_loader:
            human_data = human_batch[0].to(device)
            human_emb = model.encode(human_data)
            pred_mouse = model.decode(human_emb)
            
            human_embeddings.append(human_emb.cpu())
            predicted_mouse.append(pred_mouse.cpu())
            
        for mouse_batch in mouse_loader:
            mouse_data = mouse_batch[0].to(device)
            mouse_emb = model.encode(mouse_data)
            mouse_embeddings.append(mouse_emb.cpu())
            actual_mouse_data.append(mouse_data.cpu())
    
    # Concatenate all batches
    human_embeddings = torch.cat(human_embeddings, dim=0)
    mouse_embeddings = torch.cat(mouse_embeddings, dim=0)
    predicted_mouse = torch.cat(predicted_mouse, dim=0)
    actual_mouse = torch.cat(actual_mouse_data, dim=0)
    
    # Verify latent dimension matches config
    assert human_embeddings.shape[1] == data_info['latent_dim'], \
        f"Latent dimension mismatch: {human_embeddings.shape[1]} vs {data_info['latent_dim']}"
    
    # Compute metrics
    metrics['mmd_latent'] = compute_mmd_distance(human_embeddings, mouse_embeddings)
    metrics['mmd_output'] = compute_mmd_distance(predicted_mouse, actual_mouse)
    
    # Compute KNN enrichment
    enrichment_scores = compute_knn_enrichment(
        predicted_mouse.numpy(), 
        actual_mouse.numpy()
    )
    metrics['knn_enrichment_50'] = np.mean(enrichment_scores[:50])
    metrics['knn_enrichment_100'] = np.mean(enrichment_scores[:100])
    
    # Compute correlation metrics
    pred_mean = predicted_mouse.mean(dim=0)
    actual_mean = actual_mouse.mean(dim=0)
    metrics['correlation_means'] = F.cosine_similarity(pred_mean, actual_mean, dim=0).item()
    
    pred_std = predicted_mouse.std(dim=0)
    actual_std = actual_mouse.std(dim=0)
    metrics['correlation_stds'] = F.cosine_similarity(pred_std, actual_std, dim=0).item()
    
    # Compute per-gene correlations
    gene_correlations = {}
    for i, gene in enumerate(data_info['gene_names']):
        pred_gene = predicted_mouse[:, i]
        actual_gene = actual_mouse[:, i]
        correlation = F.cosine_similarity(pred_gene, actual_gene, dim=0).item()
        gene_correlations[gene] = correlation
    
    return metrics, gene_correlations

def main(argv):
    # Load config
    config = load_config(FLAGS.config_path)
    print("Config loaded successfully")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load and prepare data first to get input dimensions
        human_loader, mouse_loader, data_info = load_and_prepare_data(
            FLAGS.human_adata,
            FLAGS.mouse_adata,
            config
        )
        print(f"Data loaded successfully:")
        print(f"Number of genes: {data_info['n_genes']}")
        print(f"Human cells: {data_info['human_cells']}")
        print(f"Mouse cells: {data_info['mouse_cells']}")
        print(f"Latent dimension: {data_info['latent_dim']}")
        
        # Add input dimension to config
        config['data']['input_dim'] = data_info['n_genes']
        
        # Load model with updated config
        print("Loading model...")
        model = load_model(FLAGS.model_path, config, device)
        print("Model loaded successfully")
        
        # Create output directory
        outdir = Path(FLAGS.outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        
        # Save config to output directory
        with open(outdir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
    
        # Run evaluation multiple times
        all_metrics = []
        all_gene_correlations = []
        
        for rep in range(FLAGS.n_reps):
            print(f"Running evaluation repetition {rep + 1}/{FLAGS.n_reps}")
            metrics, gene_correlations = evaluate_model(
                model, 
                human_loader, 
                mouse_loader, 
                data_info,
                device
            )
            metrics['repetition'] = rep
            all_metrics.append(metrics)
            all_gene_correlations.append(gene_correlations)
        
        # Save results
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(outdir / 'evaluation_metrics.csv', index=False)
        
        # Save gene correlations
        gene_corr_df = pd.DataFrame(all_gene_correlations)
        gene_corr_df.to_csv(outdir / 'gene_correlations.csv', index=False)
        
        # Print summary statistics
        print("\nEvaluation Results (mean ± std):")
        for metric in results_df.columns:
            if metric != 'repetition':
                mean = results_df[metric].mean()
                std = results_df[metric].std()
                print(f"{metric}: {mean:.4f} ± {std:.4f}")
        
        # Print top and bottom genes by correlation
        mean_gene_corrs = gene_corr_df.mean()
        print("\nTop 10 best predicted genes:")
        print(mean_gene_corrs.nlargest(10))
        print("\nTop 10 worst predicted genes:")
        print(mean_gene_corrs.nsmallest(10))
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Print the full traceback for debuging
        import traceback
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    app.run(main)