import torch
import gymnasium as gym
import torchvision.transforms as transforms
from model import VAE, MDN_RNN, Controller
from train_controller import rollout 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    z_dim = 32
    action_dim = 3
    hidden_dim = 256
    num_gaussians = 5
    
    vae = VAE(input_shape=(3,64,64), latent_dim=z_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae.pt", map_location=device, weights_only=True))
        print("Loaded vae.pt successfully.")
    except Exception as e:
        print(f"Could not load vae.pt: {e}. Using untrained VAE.")
    vae.eval()
    
    rnn = MDN_RNN(z_dim, action_dim, hidden_dim, num_gaussians).to(device)
    try:
        rnn.load_state_dict(torch.load("rnn.pt", map_location=device, weights_only=True))
        print("Loaded rnn.pt successfully.")
    except Exception as e:
        print(f"Could not load rnn.pt: {e}. Using untrained RNN.")
    rnn.eval()
    
    controller = Controller(z_dim, hidden_dim, action_dim).to(device)
    try:
        controller.load_state_dict(torch.load("controller.pt", map_location=device, weights_only=True))
        print("Loaded controller.pt successfully.")
    except Exception as e:
        print(f"Could not load controller.pt: {e}. Using untrained Controller.")
    controller.eval()
    
    env = gym.make("CarRacing-v3", render_mode="human")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    
    print("Running episode to test generation (Rendering)...")
    reward = rollout(controller, vae, rnn, env, transform, render=True)
    
    print(f"Total Reward for this run: {reward}")
    env.close()

if __name__ == "__main__":
    main()
