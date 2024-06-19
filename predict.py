from cog import BasePredictor, Input, Path
import tempfile
import time
from meshgpt_pytorch import MeshTransformer, mesh_render
import igl
import numpy as np


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.transformer = MeshTransformer.from_pretrained(
            "MarcusLoren/MeshGPT-preview"
        )

    def save_as_obj(self, file_path):
        v, f = igl.read_triangle_mesh(file_path)
        v, f, _, _ = igl.remove_unreferenced(v, f)
        c, _ = igl.orientable_patches(f)
        f, _ = igl.orient_outward(v, f, c)
        igl.write_triangle_mesh(file_path, v, f)
        output_path = Path(tempfile.mkdtemp()) / file_path
        return output_path

    def predict(
        self,
        text: str = Input(description="Enter labels, separated by commas"),
        num_input: int = Input(description="Number of examples per input", default=1),
        num_temp: float = Input(description="Temperature (0 to 1)", default=0),
    ) -> Path:
        """Run a single prediction on the model"""
        self.transformer.eval()
        labels = [label.strip() for label in text.split(",")]

        output = []
        current_time = time.time()

        formatted_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
        )
        print(
            formatted_time,
            " Input:",
            text,
            "num_input",
            num_input,
            "num_temp",
            num_temp,
        )

        if num_input > 1:
            for label in labels:
                output.append(
                    (
                        self.transformer.generate(
                            texts=[label] * num_input, temperature=num_temp
                        )
                    )
                )
        else:
            output.append(
                (self.transformer.generate(texts=labels, temperature=num_temp))
            )
        file_name = "./mesh.obj"
        mesh_render.save_rendering(file_name, output)
        file_path = self.save_as_obj(file_name)
        return file_path


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

    print(predictor.predict("cat", 1, 0.0))
