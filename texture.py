import argparse
import os.path
import time
import torch
import torch.utils.data

from PIL import Image
from create_ply import load_scene, parse_vec3
from model import INeuTex


class MeshVertexDataset(torch.utils.data.Dataset):
    def __init__(self, vertices, faces, colors):
        # We no longer want to track gradients for the mesh components.
        self._vertices = vertices.detach()
        #self._faces = faces.detach()
        self._colors = colors.detach()

    def __len__(self):
        return self._vertices.size(0)

    def __getitem__(self, idx):
        return self._vertices[idx,:], self._colors[idx,:]


def uv_to_xy(uv, size):
    return (int(uv[0] * size[0]), int(uv[1] * size[1]))


def render_image(model, device, size=(256, 256)):
    x = torch.arange(0, size[0]).to(device=device)
    y = torch.arange(0, size[1]).to(device=device)
    u = (x + 0.5) / size[0]
    v = (y + 0.5) / size[1]
    u = u.repeat(size[1])
    v = v.repeat_interleave(size[0])
    uv = torch.stack((u, v), dim=1)
    # TODO: pass some view direction
    c = (255 * model.Ftex(uv, None)).to(dtype=torch.uint8)
    image = Image.new('RGB', size)
    for i, (u, v) in enumerate(uv):
        xy = uv_to_xy((u, v), image.size)
        rgb = c[i,:]
        image.putpixel(xy, (rgb[0], rgb[1], rgb[2]))
    return image


def main():
    render_res = (64, 64)
    num_epochs = 2
    checkpoint_rate = 4 # epochs
    batch_size = 100

    # TODO: update options/help
    p = argparse.ArgumentParser(description="Export triangle scene to PLY with per-vertex colors.")
    p.add_argument("scene_dir", type=str, help="Directory containing point_cloud_state_dict.pt")
    p.add_argument("--out", type=str, default="mesht.ply", help="Output path, default <scene_dir>/mesh.ply")
    p.add_argument("--camera", type=parse_vec3, default="0,0,0", help="Camera center x,y,z")
    p.add_argument("--degree", type=int, default=None, help="Override SH degree to evaluate")
    p.add_argument("--zup", action="store_true", help="Input is Z-up, rotate to Y-up for standard viewers")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_center = args.camera.to(device)
    vertices, faces, f_dc, _, _ = load_scene(args.scene_dir, device=device)
    SH_C0 = 0.28209479177387814
    colors = torch.clamp(SH_C0 * f_dc + 0.5, 0.0, 1.0).squeeze()
    dataset = MeshVertexDataset(vertices, faces, colors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = INeuTex(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    logfile = os.path.join(args.scene_dir, 'logt.tsv')
    with open(logfile, 'w') as f:
        f.write('Epoch\tLoss\tDuration (s)\n')
        for epoch in range(num_epochs):
            t = time.time()

            # TODO uv is always constant, check both forward and backward
            for batch, (x, y) in enumerate(dataloader):
                output = model(x)
                color = output[:,0:3]
                uv = output[:,3:5]
                loss = criterion(color, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t = time.time() - t
            f.write('{:06d}\t{:.6f}\t{:.3f}\n'.format(epoch, loss, t))
            print('epoch {:6d}\tloss {:.6f}\t{:.3f} s'.format(epoch, loss, t))
            if epoch % checkpoint_rate == 0:
                t = time.time()
                filename = os.path.join(args.scene_dir, 'ineutex-output-e{}.png'.format(epoch))
                render_image(model, device, render_res).save(filename)
                t = time.time() - t
                print('render {:6d}\t{:.3f} s'.format(epoch, t))


if __name__ == "__main__":
    main()
