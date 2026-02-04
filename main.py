import argparse
import warnings
from sklearn.cluster import KMeans
import time
import json
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from GatorTrio import Model


def train(train_loader,
          valid_loader,
          input_dim,
          num_head,
          hidden_dim,
          gcn_dim1,
          gcn_dim2,
          mlp_dim1,
          mlp_dim2,
          num_experts,
          K,
          eta,
          sigma,
          tau,
          a_1,
          a_2,
          lambda_1,
          lambda_2,
          v_drop,
          p_drop,
          lr,
          seed,
          loss1_epochs,
          epochs,
          save_model_name,
          device):
    model = Model(input_dim, num_head, hidden_dim, gcn_dim1, gcn_dim2, mlp_dim1, mlp_dim2, num_experts, 
                  K, eta, sigma, tau, a_1, a_2, lambda_1, lambda_2, v_drop, p_drop, seed).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    
    setup_seed(seed)
    train_loss = []
    valid_loss = []
    best_epoch = 0
    min_loss = 999999
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    
    start_time = time.time()
    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        for step, (batch_x, batch_y, loc) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            loc = loc.float().to(device)
            
            z_tild_emb, loss_total, loss_1 = model(batch_x, loc)
            if each_epoch < loss1_epochs:
                loss = loss_1
            else:
                loss = loss_total
            
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            batch_loss.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(np.array(batch_loss)))
        
        epoch_time = time.time() - start_time
        avg_epoch_time = epoch_time / (each_epoch + 1)
        epochs_left = epochs - (each_epoch + 1)
        est_time_left = epochs_left * avg_epoch_time

        print(f"[Epoch {each_epoch + 1}/{epochs}] "
              f"Current train_loss: {train_loss[-1]:.4f} "
              f"Elapsed: {epoch_time/60:.2f} min "
              f"ETA: {est_time_left/60:.2f} min "
              f"({epochs_left} epochs left)")
        
        with torch.no_grad():
            model.eval()
            batch_valid_loss = []

            for step, (batch_x, batch_y, loc) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                loc = loc.float().to(device)

                z_tild_emb, loss_total, loss_1 = model(batch_x, loc)
                if each_epoch < loss1_epochs:
                    loss = loss_1
                else:
                    loss = loss_total
                    
                batch_valid_loss.append(loss.cpu().detach().numpy())
                
        valid_loss.append(np.mean(np.array(batch_valid_loss)))
        cur_loss = valid_loss[-1]
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': best_epoch
                }
            torch.save(state, './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict')

    return min_loss


def test(test_loader,
          input_dim,
          num_head,
          hidden_dim,
          gcn_dim1,
          gcn_dim2,
          mlp_dim1,
          mlp_dim2,
          num_experts,
          K,
          eta,
          sigma,
          tau,
          a_1,
          a_2,
          lambda_1,
          lambda_2,
          v_drop,
          p_drop,
          seed,
          n_clusters,
          save_model_name,
          device):
    model = Model(input_dim, num_head, hidden_dim, gcn_dim1, gcn_dim2, mlp_dim1, mlp_dim2, num_experts, 
                  K, eta, sigma, tau, a_1, a_2, lambda_1, lambda_2, v_drop, p_drop, seed).to(device)
    ckpt_path = './saved_models/'+save_model_name+'_'+str(int(seed))+'_dict'
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['net'])
        
    z_test = []
    y_test = []
    for step, (batch_x, batch_y, loc) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        loc = loc.float().to(device)

        z_tild_emb, _, _ = model(batch_x, loc)
        
        z_test.append(z_tild_emb.cpu().detach().numpy())
        y_test.append(batch_y.detach().numpy())

    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_kmeans_test = kmeans.labels_

    acc, f1, nmi, ari, homo, comp, purity = evaluate(y_test, y_kmeans_test)
    result = {"ari": ari, "nmi": nmi, "acc": acc, "purity": purity, "homo": homo}

    return result

 
if __name__ == '__main__':  
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="./data/")
    parser.add_argument("--data_name", type=str, default="ST-H1")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=65)
    parser.add_argument("--num_head", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=439)
    parser.add_argument("--gcn_dim1", type=int, default=195)
    parser.add_argument("--gcn_dim2", type=int, default=167)
    parser.add_argument("--mlp_dim1", type=int, default=172)
    parser.add_argument("--mlp_dim2", type=int, default=126)
    parser.add_argument("--num_experts", type=int, default=11)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.16)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--a_1", type=float, default=0.75)
    parser.add_argument("--a_2", type=float, default=0.9)
    parser.add_argument("--lambda_1", type=float, default=0.87)
    parser.add_argument("--lambda_2", type=float, default=0.62)
    parser.add_argument("--v_drop", type=float, default=0.1)
    parser.add_argument("--p_drop", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--loss1_epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_model_name", type=str, default="model")
    
    args = parser.parse_args()
    folder_path = args.folder_path
    data_name = args.data_name
    device_n = args.device
    batch_size = args.batch_size
    num_head = args.num_head
    hidden_dim = args.hidden_dim
    gcn_dim1 = args.gcn_dim1
    gcn_dim2 = args.gcn_dim2
    mlp_dim1 = args.mlp_dim1
    mlp_dim2 = args.mlp_dim2
    num_experts = args.num_experts
    K = args.K
    eta = args.eta
    sigma = args.sigma
    tau = args.tau
    a_1 = args.a_1
    a_2 = args.a_2
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    v_drop = args.v_drop
    p_drop = args.p_drop
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    loss1_epochs = args.loss1_epochs
    save_model_name = data_name+'_'+args.save_model_name
    
    device = torch.device(f"cuda:{device_n}")
    train_set, valid_set, test_set, input_dim, n_clusters = loader_construction(folder_path + data_name + '.h5ad')
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=10)

    min_loss = train(train_loader, valid_loader, input_dim=input_dim, num_head=num_head, 
                                hidden_dim=hidden_dim, gcn_dim1=gcn_dim1, gcn_dim2=gcn_dim2, mlp_dim1=mlp_dim1, mlp_dim2=mlp_dim2, 
                                num_experts=num_experts, K=K, eta=eta, sigma=sigma, tau=tau, a_1=a_1, a_2=a_2, lambda_1=lambda_1, 
                                lambda_2=lambda_2, v_drop=v_drop, p_drop=p_drop, lr=lr, seed=seed, loss1_epochs=loss1_epochs, 
                                epochs=epochs, save_model_name=save_model_name, device=device)
    result = test(test_loader, input_dim=input_dim, num_head=num_head, 
                                hidden_dim=hidden_dim, gcn_dim1=gcn_dim1, gcn_dim2=gcn_dim2, mlp_dim1=mlp_dim1, mlp_dim2=mlp_dim2, 
                                num_experts=num_experts, K=K, eta=eta, sigma=sigma, tau=tau, a_1=a_1, a_2=a_2, lambda_1=lambda_1, 
                                lambda_2=lambda_2, v_drop=v_drop, p_drop=p_drop, seed=seed, n_clusters=n_clusters, 
                                save_model_name=save_model_name, device=device)
    result = {k: float(v) for k, v in result.items()}
    
    with open(f"./saved_results/{data_name}_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)

