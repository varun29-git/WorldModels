import torch
import numpy as np
import gymnasium as gym
import cma
import torchvision.transforms as transforms
import multiprocessing as mp

from model import VAE, MDN_RNN, Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rollout(controller, vae, rnn, env, transform, render=False):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    # initialize hidden state for RNN
    # RNN expects (num_layers, batch, hidden_size)
    hidden = (torch.zeros(1, 1, 256).to(device), torch.zeros(1, 1, 256).to(device))
    
    with torch.no_grad():
        while not (done or truncated):
            if render:
                env.render()
                
            obs_tensor = transform(obs).unsqueeze(0).to(device)
            mu, logvar = vae.encode(obs_tensor)
            z = mu # use mean vector
            
            # Controller
            h = hidden[0].squeeze(0) # (1, 256)
            action = controller(z, h)
            action_np = action.squeeze(0).cpu().numpy()
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(action_np)
            total_reward += reward
            
            # Update RNN
            # add sequence dim to z and action
            z_in = z.unsqueeze(1) # (1, 1, 32)
            a_in = action.unsqueeze(1) # (1, 1, 3)
            _, _, _, hidden = rnn(z_in, a_in, hidden)
            
    return total_reward

def evaluate_weights(weights, controller, vae, rnn, env, transform):
    torch.nn.utils.vector_to_parameters(torch.tensor(weights, dtype=torch.float32).to(device), controller.parameters())
    return -rollout(controller, vae, rnn, env, transform)

def main():
    z_dim = 32
    action_dim = 3
    hidden_dim = 256
    num_gaussians = 5
    
    vae = VAE(input_shape=(3,64,64), latent_dim=z_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae.pt", map_location=device, weights_only=True))
        print("Loaded vae.pt")
    except Exception as e:
        print(f"Note: vae.pt not loaded ({e}), using untrained VAE")
    vae.eval()
    
    rnn = MDN_RNN(z_dim, action_dim, hidden_dim, num_gaussians).to(device)
    try:
        rnn.load_state_dict(torch.load("rnn.pt", map_location=device, weights_only=True))
        print("Loaded rnn.pt")
    except Exception as e:
        print(f"Note: rnn.pt not loaded ({e}), using untrained RNN")
    rnn.eval()
    
    controller = Controller(z_dim, hidden_dim, action_dim).to(device)
    
    initial_weights = torch.nn.utils.parameters_to_vector(controller.parameters()).detach().cpu().numpy()
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    
    es = cma.CMAEvolutionStrategy(initial_weights, 0.1, {'popsize': 16}) 
    
    generation = 0
    max_generations = 50
    print("Starting CMA-ES optimization for Controller...")
    
    while not es.stop() and generation < max_generations:
        solutions = es.ask()
        
        # Eval solutions (synchronous for simplicity)
        fitnesses = [evaluate_weights(w, controller, vae, rnn, env, transform) for w in solutions]
        
        es.tell(solutions, fitnesses)
        es.disp()
        
        best_weights = es.result.xbest
        print(f"Generation {generation} | Best Reward: {-es.result.fbest:.2f}")
        
        torch.nn.utils.vector_to_parameters(torch.tensor(best_weights, dtype=torch.float32).to(device), controller.parameters())
        torch.save(controller.state_dict(), "controller.pt")
        
        generation += 1

    env.close()
    print("Optimization finished.")

if __name__ == "__main__":
    main()
