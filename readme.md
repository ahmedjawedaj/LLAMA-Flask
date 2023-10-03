# LLaMA Text Generation API Readme

Welcome to LLaMA Text Generation API! This API is implemented in Python using Flask and utilizes a pre-trained LLaMA model for generating text based on user input.

## Table of Contents

- [Setting Up Virtual Environment](#setting-up-virtual-environment)
- [Installing Requirements](#installing-requirements)
- [Choosing a GPU](#choosing-a-gpu)
- [Running API with Torch Command](#running-api-with-torch-command)
- [Running API using Gunicorn](#running-api-using-gunicorn)
- [API Usage with cURL](#api-usage-with-curl)
- [Request/Response Objects](#requestresponse-objects)
- [Using Postman](#using-postman)

### Setting Up Virtual Environment

1. **Install Virtual Environment:**
   
   Ensure that Python 3 and `pip` are installed and then run:
   ```bash
   pip install virtualenv
   ```

2. **Create Virtual Environment:**

   Navigate to the project directory and run:
   ```bash
   virtualenv venv
   ```
   
3. **Activate Virtual Environment:**

   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```
   
## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-llama
cd lit-llama
```

install dependencies

```bash
pip install -r requirements.txt
```

You are all set! 🎉

&nbsp;

### Choosing a GPU

- **LLaMA 7B and 13B Models:** Sufficiently run on an A500 with 24GB VRAM.
- **LLaMA 30B Model:** Requires a more powerful GPU such as A40 with 48GB VRAM.

### Running API with Torch Command

You should navigate to the project directory and run:
```bash
python app.py
```
The API will be hosted on `http://0.0.0.0:5000/complete`.

### Running API using Gunicorn

1. **Install Gunicorn:**
   Ensure the virtual environment is activated and then run:
   ```bash
   pip install gunicorn
   ```

2. **Run API:**
   Use gunicorn to serve the Flask app:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
   
3. **Set Up Gunicorn as a Service:**
   - Create a gunicorn systemd service file:
     ```bash
     sudo nano /etc/systemd/system/llama-api.service
     ```
   - Add the following content and adjust paths accordingly:
     ```
     [Unit]
     Description=Gunicorn instance to serve LLaMA API
     After=network.target

     [Service]
     User=your_user
     Group=www-data
     WorkingDirectory=/path/to/your/project
     Environment="PATH=/path/to/your/project/venv/bin"
     ExecStart=/path/to/your/project/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
     
     [Install]
     WantedBy=multi-user.target
     ```
   - Start and enable the gunicorn service:
     ```bash
     sudo systemctl start llama-api
     sudo systemctl enable llama-api
     ```

### API Usage with cURL

Example cURL request:
```bash
curl -X POST http://0.0.0.0:5000/complete \
-H "Content-Type: application/json" \
-d '{"text": "Once upon a time,", "top_p": 0.9, "top_k": 50, "temperature": 0.8, "length": 30}'
```

Example response:
```json
{
   "completion":{
      "generation_time":"0.8679995536804199s",
      "text":["Once upon a time, the kingdom was ruled by a wise and just king..."]
   }
}
```

### Request/Response Objects

- **Request:**

  - `text`: The input text (string).
  - `top_p`: Probability for nucleus sampling (float).
  - `top_k`: The number of top most probable tokens to consider (integer).
  - `temperature`: Controls the randomness of the sampling process (float).
  - `length`: The number of new tokens to generate (integer).

- **Response:**

  - `text`: The generated text based on the input (string).
  - `generation_time`: Time taken to generate the text (string, formatted as seconds).

### Using Postman

1. **Set Up Postman:**
   Download and install Postman from [Postman's official site](https://www.postman.com/).
   
2. **Send Request:**
   - Set the request type to `POST`.
   - Enter the request URL: `http://0.0.0.0:5000/complete`.
   - Navigate to the "Body" tab, select "raw" and "JSON (application/json)".
   - Enter the JSON payload:
     ```json
     {
        "text": "Once upon a time,",
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.8,
        "length": 30
     }
     ```
   - Click "Send" and view the API's response in the section below.

And that concludes our README guide! Feel free to adapt this guide as per additional requirements for your API.
