from cog import BasePredictor, Input, Path
import tempfile
import time
from meshgpt_pytorch import MeshTransformer, mesh_render


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.transformer = MeshTransformer.from_pretrained(
            "MarcusLoren/MeshGPT-preview"
        )

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
        file_path = Path(tempfile.mkdtemp()) / file_name
        mesh_render.save_rendering(str(file_path), output)
        return file_path
