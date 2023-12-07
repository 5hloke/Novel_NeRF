import torch
import os
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import imageio
from PIL import Image
from skimage.transform import resize
# Make sure to switch runtime to the GPU
import gc
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
print(device)
gc.collect()
torch.cuda.empty_cache()

DATA = np.load('tiny_nerf_data.npz')
class FrameManager:
    def __init__(self):
        self.test_frames = []
        self.train_frames = []
        self.val_frames = []
        self.cam_angle = 0
        self.f = None
        self.H = None
        self.W = None


    def read_frames(self, path):
        img = path['images']
        poses = path['poses']
        self.f = path['focal']
                # img = resize(img, (100, 100)) ### image scaled down to test
        self.H, self.W = img.shape[0], img.shape[1]
    
        # new_frame = Frame(img, np.array(frame['transform_matrix']), self.f)
        for i in range(len(img)):
            new_frame = Frame(img[i], poses[i], self.f)
            self.train_frames.append(new_frame)


class Frame:
    def __init__(self, image, pose, f):
        self.img = image
        self.pose = pose
        self.H, self.W = image.shape[0], image.shape[1]
        self.f = torch.from_numpy(f).to('cuda')
        self.samples = None
        self.rays_o = None
        self.rays_d = None
        self.depth_values = None

    def copy_to_device(self, device):
        self.img = torch.from_numpy(self.img).to(device)
        self.pose = torch.from_numpy(self.pose).to(device)

    def make_tensors(self):
        if not(torch.is_tensor(self.img)):
            self.img = torch.from_numpy(self.img)
        if not(torch.is_tensor(self.pose)):
            self.pose = torch.from_numpy(self.pose)
        self.img = self.img.to(torch.float64)
        self.pose = self.pose.to(torch.float64)
    

    # function to get the rays from the image through every pixel of the Camera (Using Pytorch) on GPU
    # Assuming a pinhole camera model
    def get_rays(self, device):
        # self.copy_to_device(device)
        self.make_tensors()
        i, j = torch.meshgrid(torch.arange(self.H).to(device), torch.arange(self.W).to(device), indexing='ij')
        # i, j = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing = 'ij')
        i, j = i.transpose(-1, -2), j.transpose(-1, -2)
        dirs = torch.stack([(i-self.W*0.5)/self.f, -(j-self.H*0.5)/self.f, -torch.ones_like(i)], -1)
        if (self.pose.device != device):
            self.pose = self.pose.to(device)
        # print(dirs.device)
        rays_d = torch.sum(dirs[..., None, :] * self.pose[:3, :3], -1)
        rays_o = torch.broadcast_to(self.pose[:3, -1], rays_d.shape)
        # self.pose = self.pose.to("cpu")
        self.rays_o = rays_o
        self.rays_d = rays_d
        
        # del i
        # del j
        return rays_o.view([-1, 3]), rays_d.view([-1, 3])
    
def sample_frame(frames, num_samples, near, far, dev = 'cuda'):

    sample_space = torch.linspace(0., 1., num_samples, device=dev)
    rays_o = frames.rays_o.reshape([-1, 3])
    rays_d = frames.rays_d.reshape([-1, 3])
    # sample_space = torch.linspace(0., 1., num_samples)
    depth = near*(1.-sample_space) + far*sample_space
    mid_depth = (depth[1:] + depth[:-1])/2
    rand_sampling = torch.rand([num_samples], device=dev)
    # rand_sampling = torch.rand([num_samples])
    upper_sample = torch.cat([mid_depth, depth[-1:]], dim=-1)
    lower_sample = torch.cat([depth[:1], mid_depth], dim=-1)
    depth_value = lower_sample + rand_sampling * (upper_sample - lower_sample)
    depth_value = depth_value.expand(list(rays_o.shape[:-1]) +[num_samples])
    #pts are the points on the ray in the format (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * depth_value[..., :, None]
    frames.samples = pts
    frames.depth_values = depth_value
    # del rand_sampling
    # del sample_space
    return pts, depth_value

class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_layer = nn.Linear(60, 256, dtype=torch.float64)
        self.hidden_layer_block_1 = nn.ModuleList([nn.Linear(256, 256, dtype=torch.float64) for i in range(4)])
        self.skip_connection_layer = nn.Linear(316,256, dtype=torch.float64)
        self.hidden_layer_block_2 = nn.ModuleList([nn.Linear(256, 256, dtype=torch.float64) for i in range(2)])
        self.density_output_layer = nn.Linear(256, 256, dtype=torch.float64)
        self.rgb_layer = nn.Linear(256, 256, dtype=torch.float64)
        self.last_layer = nn.Linear(280, 128, dtype=torch.float64)
        self.rgb_output = nn.Linear(128, 3, dtype=torch.float64)


    def forward(self, position, direction):

        # print(position.size())
        # print(direction.size())

        encoded_position, encoded_direction = self.positional_encoding(position, direction) # should return shape of (# of samples, 60) and (# of samples, 24)
        # print(encoded_position.dtype)
        # print(encoded_direction.dtype)
        # print(encoded_position.size())
        # print(encoded_direction.size())
        # input_feature_origin = encoded_position.clone().to(device)
        x = nn.functional.relu(self.input_layer(encoded_position))
        # print(x.dtype)
        for layer in self.hidden_layer_block_1:
            x = nn.functional.relu(layer(x))
        # print(x.dtype)
        skip_connection = torch.cat((x, encoded_position), dim=-1)
        x = nn.functional.relu(self.skip_connection_layer(skip_connection))
        # print(x.dtype)
        for layer in self.hidden_layer_block_2:
            x = nn.functional.relu(layer(x))
        # print(x.dtype)
        dens_x = nn.functional.relu(self.density_output_layer(x))
        # print(x.dtype)
        # print(x.size())
        density = dens_x[:,-1]
        x = self.rgb_layer(x)
        x = torch.cat([x, encoded_direction], dim=-1)
        # print(direction_connection.size())
        x = nn.functional.relu(self.last_layer(x))
        # print(x.dtype)
        color = torch.sigmoid(self.rgb_output(x))
        
        del encoded_position
        del encoded_direction
        # del direction_connection
        
        return color, density

    def positional_encoding(self, position, direction, L_P = 10, L_D =4):
        # direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        direction = direction[:, None, ...].expand(position.shape).reshape((-1, 3))
        position = position.reshape([-1, 3])
        
        encoded_position = []
        encoded_direction = []

        for i in range (L_P):
            encoded_position.append(torch.sin(2**i * np.pi * position))
            encoded_position.append(torch.cos(2**i * np.pi * position))

        for i in range (L_D):
            encoded_direction.append(torch.sin(2**i * np.pi * direction))
            encoded_direction.append(torch.cos(2**i * np.pi * direction))

        # print(torch.cat(encoded_position, dim=1).size())
        # print(encoded_direction)

        return torch.cat(encoded_position, dim=1), torch.cat(encoded_direction, dim=1)




# assuming that points is a list of size (num_of_rays, num_of_samples, 6) so 3 points for position, 3 for direction for every sample on a ray
# render_rays will query the network at each sample point to determine the color and density and that point. We then use volume rendering to determine the final color.
# dt is distance between adjacent samples; prob just near-far / num of samples for uniform sampling
def render_rays(model, frame):
    gc.collect()
    torch.cuda.empty_cache()
    # dt = torch.ones(points.size()[0])/points.size()[0] # just for testing
    positions = frame.samples
    directions = frame.rays_d.reshape([-1, 3])
    color, density = model(positions, directions) # color should be (# of samples * # of rays, 3) density should be (# of samples * # of rays, 1)
    # rendering begins here
    dt = frame.depth_values[..., 1:] - frame.depth_values[..., :-1]
    dt = torch.cat([dt, 1e20*torch.ones_like(dt[..., :1])], dim = -1).to(device) #torch.Size([800, 800, 8])
    
    color = torch.reshape(color, frame.samples.shape)#.to(device)
    density = torch.reshape(density, (frame.samples.shape[0], frame.samples.shape[1]))#.to(device)
    alpha = 1 - torch.exp(-density*dt)
    transmittance = torch.cumprod((1-alpha), dim=1)
    transmittance = torch.roll(transmittance, 1, -1)
    transmittance[..., 0] = 1

    # print(transmittance.shape)
    # print(alpha.shape)
    # print(color.shape)
    rgb = torch.sum(alpha[:,:,None] * transmittance[:,:,None] * color, dim=-2)
    del dt
    return rgb


def train(model, epochs, data, learning_rate = 5e-4):
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    # print()
    index_test = 101
    count = 0
    for fr in data.train_frames:
        # print("Getting rays and sampling")
        fr.get_rays(device)
        # sample_frame(fr, 100, 2, 6, dev = 'cuda:0')
        # print(count)
        count +=1
    plt.imshow(data.train_frames[index_test].img)
    plt.savefig(f'test_img_{index_test}_original.png')
    plt.close()

    ls = []
    for i in range(epochs):
      model.train()
      print("Epoch number", i)
      index = np.random.randint(len(data.train_frames))
      fr = data.train_frames[index]
      sample_frame(fr, 64, 2, 6, dev = 'cuda:0')
      # print(fr.img.shape)
      # fr.get_rays(device)
      # sample_frame(fr, 8, 1, 10, dev = 'cuda:0)
      predicted = render_rays(model, fr)
      # print(predicted.device)
      actual = fr.img.to(device)
      # print(fr.img.device)
      loss = torch.nn.functional.mse_loss(predicted.reshape([actual.shape[0], actual.shape[1], actual.shape[2]]), actual[:, :, :]) # fr.img is (800,800,4)
      print("Loss:", loss.item())
      ls.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i % 25 == 0:
        torch.save(model.state_dict(), 'nerf_2.pt')
        test(model, data, index_test,i, loss) ## test on same test image each epoch
    
    return loss

def test(model, data, index_test, epoch, loss):
    model.eval()
    fr = data.train_frames[index_test]
    # img = fr.img.to(device)
    # fr.get_rays(device)
    sample_frame(fr, 64, 2, 6, dev = 'cuda:0')
    predicted = render_rays(model, fr)
    predicted = predicted.detach().cpu().numpy()
    plt.imshow(predicted.reshape([100, 100, 3]))
    plt.savefig(f'test_img_{index_test}_epoch_{epoch}.png')
    plt.close()
    del predicted

if __name__ == '__main__':
    frameManager = FrameManager()
    frameManager.read_frames(DATA)
    gc.collect()
    torch.cuda.empty_cache()
    model = Model()
    state_dict = torch.load('nerf_2.pt')
    # for key in list(state_dict.keys()):
    #     state_dict[key.replace('rgb_filters.weight', 'rgb_layer.weight'). replace('rgb_filters.bias', 'rgb_layer.bias')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    loss = train(model, 10000, frameManager)
# frameManager.read_frames('drive/MyDrive/EECS 504 Final Project/NOVEL_NERF/Data/lego')
