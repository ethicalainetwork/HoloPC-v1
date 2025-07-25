<h1 align="center" id="title">HoloPC</h1>

<p align="center"><img src="https://socialify.git.ci/ethicalainetwork/HoloPC-v1/image?description=1&amp;font=Raleway&amp;forks=1&amp;issues=1&amp;name=1&amp;owner=1&amp;pattern=Solid&amp;pulls=1&amp;stargazers=1&amp;theme=Auto" alt="project-image"></p>

<p id="description">
HoloPC is an open-source framework designed to transform any device into a fully functional holographic computer using Pepper’s Ghost projection. More than just an illusion, HoloPC enables a projected interface enriched with AI capabilities, real-time translation, transcription, and multi-agent interaction.
</p>

Whether you’re working with a laptop, tablet, or custom setup, HoloPC can display floating holographic UIs, process voice commands, and seamlessly integrate with GenOS, NLA models, or your own AI agents. Built to be browser-compatible and hardware-flexible, it supports:
- AI-powered translation & transcription
- Voice and gesture-based interaction
- Extendable plugin system
- Easy integration with autonomous agents and LLMs
- Native compatibility with GenOS environments
- Modular interface projection with dynamic layout generation

**🔧 Note**: HoloPC projects visuals that require reflection on a translucent surface for optimal effect. Full setup instructions, including how to build the holographic mirror, are available in the 📕 Building the Mirror section.

---

<h2 align="center">Project Screenshots:</h2>

<p align="center"><img src="https://i.ibb.co/dQJnFbP/displayed-images.png" alt="project-screenshot" width="590"></p>

---

<h2 align="center">🧐 Features</h2>

HoloPC combines innovative display techniques with powerful backend processing to deliver a unique holographic experience. Here are some of its key features:

* **Holographic Illusion:** Create the mesmerizing illusion of a hologram on any screen by leveraging the classic Pepper's Ghost Effect.
* **Custom Media Projection:** Easily upload your own images, GIFs, or even real-time streams to be projected as holographic content.
* **Customizable Display Structures:** Adjust the size and shape of the on-screen projection structure precisely to fit your physical mirror setup, ensuring optimal illusion quality.
* **Advanced Speech-to-Text Integration:** Process audio inputs for live transcription and translation, enabling dynamic content generation that responds to spoken words.
    * The project has evolved from an initial pipeline of `Speech → Server → Text-to-Image → GIF → Blob → Display`.
    * The current, more efficient pipeline is `Speech → Server → Direct Text Polling → Canvas Rendering`, offering improved responsiveness.
* **Cross-Device & Cross-Browser Compatibility:** Designed to be fully responsive, HoloPC works seamlessly across various devices (smartphones, tablets, desktop computers) and all major web browsers.
* **User-Friendly Interface:** The application is intuitive and straightforward to use; simply upload your media, and HoloPC handles the complex projection mechanics.
* **Open-Source & Extensible:** HoloPC is an open-source project, making it freely available for use, modification, and community contributions.

---

<h2 align="center">🚀 Getting Started</h2>

You can experience HoloPC in several ways: by visiting the hosted website, by running it using Docker for simplicity, or by building the source code yourself for local development and greater control.

### Hosted Version

For the quickest way to try HoloPC, access the live, hosted version at:
[https://holopc.nla.la](https://holopc.nla.la)

### Using Docker (Recommended for Quick Setup)

The easiest way to get HoloPC up and running is by using Docker. This method encapsulates all dependencies and services within containers, simplifying the setup process.

1.  **Ensure Docker is Installed:** Make sure you have Docker Desktop (or Docker Engine and Docker Compose) installed on your system. You can download it from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ethicalainetwork/HoloPC-v1.git](https://github.com/ethicalainetwork/HoloPC-v1.git)
    cd HoloPC-v1
    ```

3.  **Launch the Project:**
    From the project's root directory, execute:
    ```bash
    docker compose up -d
    ```
    This command will build (if necessary) and start all required services in the background.

4.  **Access HoloPC:**
    Once the containers are running, open your web browser and navigate to:
    ```
    http://localhost:3000
    ```

### Building from Source (For Local Development Without Docker)

If you prefer to run HoloPC without Docker, you'll need to install the dependencies and manage the services manually.

#### Prerequisites

Before you begin, ensure you have the following software installed on your system:

* **Git:** Essential for cloning the project repository.
* **Node.js (v16+ recommended):** Includes `npm` (Node Package Manager), which is used to manage the Angular frontend and Express.js server dependencies. Download from [nodejs.org/en/download](https://nodejs.org/en/download).
* **Python 3 (v3.8+ recommended):** Required for the Whisper streaming backend.

#### Installation and Build Steps

1.  **Clone the repository:**
    Open your terminal or command prompt and execute:
    ```bash
    git clone [https://github.com/ethicalainetwork/HoloPC-v1.git](https://github.com/ethicalainetwork/HoloPC-v1.git)
    cd HoloPC-v1
    ```

2.  **Install Node.js Dependencies and Build Frontend:**
    From the project's root directory:
    ```bash
    npm install
    npm run build
    ```

3.  **Install Python Dependencies:**
    Navigate into the `server/` directory, create and activate a Python virtual environment, then install the necessary Python libraries:
    ```bash
    cd server/
    python3 -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    pip install librosa soundfile faster-whisper websockets
    deactivate # Deactivate the virtual environment
    cd .. # Go back to the project root
    ```
    *Note: `faster-whisper` is the recommended backend for superior performance. If you plan to utilize GPU acceleration, please consult the `faster-whisper` documentation for specific NVIDIA driver and library requirements (e.g., CUDNN 8.5.0, CUDA 11.7).*

4. **(Optional) Configure Whisper Backend (Non-Docker):**
    The project is set up to use `faster-whisper` by default. If you need to switch to an alternative backend:
    * **`whisper-timestamped` (alternative, but slower):**
        ```bash
        pip install git+[https://github.com/linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)
        ```
    * **OpenAI API (requires an API key and incurs costs):**
        ```bash
        pip install openai
        export OPENAI_API_KEY=sk-your_actual_api_key # Replace with your secret API key
        ```
        *When using the OpenAI API backend, be mindful of your usage and potential costs, as audio fragments are processed multiple times. Setting a higher `chunk-size` can help reduce expenses.*

#### Running in Development Mode (Non-Docker)

After installing all dependencies as described above, you can launch HoloPC using the provided script:

1.  **Start all services:**
    From the project's root directory, execute:
    ```bash
    sh newfullrun.sh
    ```
    This script will launch the Angular frontend, Express.js server, and the Python Whisper backend, each within its own `screen` session.

2.  **Verify and manage running services:**
    To see a list of active `screen` sessions:
    ```bash
    screen -ls
    ```
    To attach to a specific service's console (e.g., for logs or debugging):
    ```bash
    screen -r fe      # Attach to the Angular frontend session
    screen -r whisper # Attach to the Whisper Python backend session
    screen -r js      # Attach to the Express.js server session
    ```
    *To detach from a `screen` session without stopping it, press `Ctrl+A`, then `D`.*

3.  **Access HoloPC:**
    Once all services are up and running, open your web browser and navigate to:
    * **Frontend (Angular):** `http://localhost:4200`
    * **Web Application (for audio input):** `http://localhost:3333` (or `http://localhost:4200` depending on your specific configuration)

#### Manual Service Startup (Advanced Users/Debugging - Non-Docker)

If you prefer granular control or need to debug individual components, you can start each service manually:

* **Angular Frontend:**
    Navigate to the project's root and run:
    ```bash
    npm start
    ```
    This typically serves the frontend on `http://localhost:4200`.

* **Express.js Server:**
    Navigate to the `server` directory and run:
    ```bash
    node server.js
    ```
    This server typically listens on `http://localhost:3000`.

* **Whisper Streaming Backend (Python):**
    Navigate to the project's root. Ensure your Python virtual environment is activated, then run:
    ```bash
    cd server/
    source venv/bin/activate # Activate venv
    python3 whisper_streaming/optimized_streaming_simple.py --model tiny.en
    # When done: deactivate
    cd ..
    ```
    This backend typically listens on `http://localhost:8765`.
    *You can specify a different Whisper model (e.g., `--model medium.en`) based on your system's available VRAM and desired accuracy. Refer to the "Whisper Model Sizes and Performance" table below.*

---

<h2 align="center">⚙️ Whisper Model Sizes and Performance</h2>

The choice of Whisper model for the speech-to-text backend significantly impacts both the required VRAM (Video RAM) and processing speed. Select a model that best suits your hardware capabilities and performance needs:

| Size   | Parameters | English-only Model | Multilingual Model | Required VRAM | Relative Speed |
| :----- | :--------- | :----------------- | :----------------- | :------------ | :------------- |
| `tiny` | 39 M       | `tiny.en`          | `tiny`             | ~1 GB         | ~10x           |
| `base` | 74 M       | `base.en`          | `base`             | ~1 GB         | ~7x            |
| `small`| 244 M      | `small.en`         | `small`            | ~2 GB         | ~4x            |
| `medium`| 769 M     | `medium.en`        | `medium`           | ~5 GB         | ~2x            |
| `large`| 1550 M     | N/A                | `large`            | ~10 GB        | 1x             |
| `turbo`| 809 M      | N/A                | `turbo`            | ~6 GB         | ~8x            |

*(Note: The `turbo` model is a `faster-whisper` specific optimization, generally providing a good balance of size and speed.)*

---

<h2 align="center">🧱 Building the Mirror</h2>

To create the holographic projection effect using HoloPC, you’ll need a simple yet precise physical structure made of 4 trapezoidal pieces of acrylic glass. This forms a reflective pyramid that sits on your device’s screen and reflects the rendered images to appear as floating holograms in mid-air.

<br>

### 📐 What You Need:
- Material: Clear acrylic glass (also known as plexiglass or PMMA), 2–3mm thick
- Quantity: 4 identical isosceles trapezoids


### 📏 Recommended Dimensions (per trapezoid):

|Top Width  |   Bottom Width |    Height |   Angle |
|-----------|----------------|-----------|---------|
| 1.5 cm    |  6 cm          |    3.5 cm |   ~45°  |


⚠️ Dimensions can be scaled depending on your device and desired hologram size. Larger screens may need larger trapezoids—just keep all four identical.

---
<h2 align="center">🧰 🛠️ Assembly Steps</h2>

    
1. **Cut the Acrylic**\
Use a precision laser cutter or fine saw to cut 4 identical trapezoidal shapes from your acrylic sheet.
    
2. **Clean the Surfaces**\
Remove any protective film and wipe the acrylic clean with a soft cloth to ensure maximum clarity.
    
3. **Form the Pyramid**\
Arrange the 4 pieces with their smaller edges facing upward, joining the slanted sides together to form a pyramid with an open top and bottom.
    
4. **Bond the Edges**\
Use clear acrylic glue or transparent double-sided tape to carefully bond the sides together. Ensure the angles are symmetrical and all joints are clean.
    
5. **Place on Screen**\
Invert the pyramid (small edge on top) and place it centered on your device screen where HoloPC is running. The projection will now reflect from all four sides, creating a floating image above the base.

<br />

Here is an example image of a 4-sided mirror setup:
<br />
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/e/e2/Pyramid_holographic_3D_holographic_projection_phone_projector_3D_holographic_projection_3D_mobile_phone_naked_eye_3D_pyramid.jpg" alt="image of a mirror" width="300px"></p>
<p align="center">&copy; Image licensed by EthicalAI under the [Creative Commons Attribution-Share Alike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/deed.en), uploaded on wikipedia.org.</p>

<br/>

**🔧 Tips for Best Results:**
- Use a dark background in your environment to enhance the floating effect.
- Angle accuracy is crucial — all sides must match for the illusion to align.
- You can also 3D print a base to hold the mirror stable, especially for mobile setups.

---

<h2 align="center">💻 Built With</h2>

HoloPC leverages a modern stack of technologies to deliver its functionality:

* **Angular 15:** Provides the robust and dynamic framework for the frontend user interface.
* **TailwindCSS:** Used for efficient and highly customizable styling, enabling rapid UI development.
* **Node.js:** Serves as the runtime environment for both the frontend's build process and the Express.js backend server.
* **Python 3:** Powers the advanced Whisper streaming backend, handling complex speech-to-text operations.
* **Express.js:** The fast, unopinionated, minimalist web framework used for the project's backend server.
* **Whisper (by OpenAI), Faster-Whisper, Whisper-Timestamped:** Key technologies for cutting-edge speech transcription and translation capabilities.
* **WebSockets:** Enables real-time, bi-directional communication between the frontend and backend services for smooth interaction.

---

<h2 align="center">🛡️ License</h2>

This project is licensed under the MIT License. You can find the full license text in the [LICENSE](https://github.com/ethicalainetwork/HoloPC-v1/blob/main/LICENSE) file within the repository.