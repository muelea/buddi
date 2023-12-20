# open3d renderer and camera coordinate system
# https://pytorch3d.org/docs/renderer_getting_started
from pytorch3d.renderer import TexturesVertex
import torch

class Texturer():
    def __init__(
        self,
        device = 'cpu'
    ) -> None:
        super().__init__()

        self.device = device
        self.create_colors()
        self.create_num_verts()

    def create_colors(self):

        self.colors = {
            'gray': [0.7, 0.7, 0.7],
            'red': [1.0, 0.0, 0.0],
            'blue': [0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0],
            'yellow': [1.0, 1.0, 0.0],
            'paper_blue': [0.9803921568627451, 0.19607843137254902, 0.8235294117647058], #[50 / 255, 210 / 255, 250 / 255]
            'paper_red': [0.9803921568627451, 1.0, 0.36470588235294116] #[255 / 255, 93 / 255, 81 / 255]
        }

        for streng in range(1, 11, 1):
            sf = streng / 10
            self.colors[f'light_blue{streng}'] = [1.0, sf, sf]
            self.colors[f'light_green{streng}'] = [sf, 1.0, sf]
            self.colors[f'light_red{streng}'] = [sf, sf, 1.0]

            self.colors[f'light_yellow{streng}'] = [sf, 1.0, 1.0]
            self.colors[f'light_pink{streng}'] = [1.0, sf, 1.0]
            self.colors[f'light_turquoise{streng}'] = [1.0, 1.0, sf]

            self.colors[f'light_orange{streng}'] = [sf, 0.3+sf*0.3, 1.0]
            self.colors[f'light_aqua{streng}'] = [1.0, 0.3+sf*0.3, sf]
            self.colors[f'light_ggreen{streng}'] = [sf, 1.0, 0.3+sf*0.3]            
    
    def create_num_verts(self):
        self.num_vertices = {
            'smpl': 6890,
            'smplh': 6890,
            'smplx': 10475,
            'smplxa': 10475,
        }
    
    def get_color(self, color):
        return self.colors[color]

    def get_num_vertices(self, body_model):
        return self.num_vertices[body_model]

    def create_texture(self, vertices, color='gray'):
        """
        Create texture for a batch of vertices. 
        Vertex dimensions are expectes to be (batch_size, num_vertices, 3).
        """
        color_rgb = self.get_color(color)
        verts_rgb = torch.ones_like(vertices).to(self.device)
        for i in [0,1,2]:
            verts_rgb[:, :, i] *= color_rgb[i]
        textures = TexturesVertex(verts_features=verts_rgb)
        return textures
    
    def quick_texture(self, vertices=None, batch_size=2, body_model='smplx', colors=['blue', 'red']):
        """
        Create texture for meshes. If vertices are provided, batch size and number of vertices is taken from there.
        Otherwise, these two parameters need to be provided.
        """
        if vertices is not None:
            batch_size, num_vertices, _ = vertices.shape
        else:
            num_vertices = self.get_num_vertices(body_model)
    
        dim = (batch_size, num_vertices, 3)
        verts_rgb = torch.ones(dim).to(self.device)

        for idx, color in enumerate(colors):
            verts_rgb_col = self.get_color(color) * num_vertices
            verts_rgb_col = torch.tensor(verts_rgb_col) \
                .reshape(-1,3).to(self.device)
            verts_rgb[idx, :, :] = verts_rgb_col

        textures = TexturesVertex(verts_features=verts_rgb)

        return textures
