import asyncio
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
from perchance import ImageGenerator

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    shape = data.get("shape", "square")
    guidance_scale = data.get("guidance_scale", 7.0)
    negative_prompt = data.get("negative_prompt")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(
            _generate_image(prompt, shape, guidance_scale, negative_prompt)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()


async def _generate_image(prompt, shape, guidance_scale, negative_prompt):
    async with ImageGenerator() as gen:
        result = await gen.image(
            prompt,
            shape=shape,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )
        binary = await result.download()
        image = Image.open(binary)

        buffered = io.BytesIO()
        img_format = result.file_extension.upper()
        if img_format == "JPG":
            img_format = "JPEG"
        image.save(buffered, format=img_format)

        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return {
            "image": img_base64,
            "width": result.width,
            "height": result.height,
            "seed": result.seed,
            "format": result.file_extension,
        }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
