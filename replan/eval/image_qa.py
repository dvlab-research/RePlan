import argparse
import vertexai
import os
import json
import base64
import mimetypes
from pathlib import Path
from typing import Union, List, Optional, Literal
import io
import yaml
import time
from PIL import Image

# Try importing new SDKs
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
except ImportError:
    vertexai = None

class RetryableError(Exception):
    pass


class GeminiImageQA:
    def __init__(self, 
                 api_config_path: Optional[str] = None,
                 project_id: Optional[str] = None, 
                 location: Optional[str] = None, 
                 credential_file: Optional[str] = None,
                 model_name: str = None, 
                 provider: Optional[Literal["vertexai", "genai", "openai"]] = None,
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize GeminiImageQA instance.

        Args:
            project_id (str): Your Google Cloud project ID (for vertexai/genai).
            location (str): Google Cloud location for Vertex AI (e.g., "us-central1").
            model_name (str, optional): Model name. Defaults to "gemini-2.5-pro".
            provider (str, optional): Backend provider ("vertexai", "genai", "openai"). Defaults to "vertexai" for compatibility.
            api_key (str, optional): API Key (for openai or genai non-vertex mode).
            base_url (str, optional): Base URL (for openai).
        """
        if api_config_path:
            with open(api_config_path, 'r') as f:
                api_config = yaml.safe_load(f)
        else:
            default_config_path = os.path.join(os.path.dirname(__file__), 'api_config.yaml')
            with open(default_config_path, 'r') as f:
                api_config = yaml.safe_load(f)
        
            self.provider = provider if provider else api_config.get('provider', None)
            self.model_name = model_name if model_name else api_config.get('model_name', None)
            project_id = project_id if project_id else api_config.get('gemini_project_id', None)
            location = location if location else api_config.get('gemini_location', None)
            credential_file = credential_file if credential_file else api_config.get('credential_file', None)
            gemini_api_key = gemini_api_key if gemini_api_key else api_config.get('gemini_api_key', None)
            openai_api_key = openai_api_key if openai_api_key else api_config.get('openai_api_key', None)
            base_url = base_url if base_url else api_config.get('base_url', None)

        if credential_file:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_file
        
        if self.provider == "vertexai":
            self.model = self._initialize_vertexai(project_id, location, model_name)
        elif self.provider == "genai":
            self.client = self._initialize_genai(project_id, location, gemini_api_key)
        elif self.provider == "openai":
            self.client = self._initialize_openai(openai_api_key, base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _initialize_vertexai(self, project_id: str, location: str, model_name: str):
        print(f"Initializing Vertex AI (Legacy) for project: {project_id}, location: {location}...")
        try:
            vertexai.init(project=project_id, location=location)
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            raise ConnectionError(f"Vertex AI initialization failed: {e}")

        print(f"Loading generative model: {model_name}...")
        try:
            return GenerativeModel(model_name)
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            raise ValueError(f"Model loading failed for {model_name}: {e}")

    def _initialize_genai(self, project_id: str, location: str, api_key: str = None):
        if not genai:
            raise ImportError("google-genai package is not installed. Please run `pip install google-genai`")
        
        print(f"Initializing Google GenAI Client...")
        try:
            if api_key:
                print("Using API Key for GenAI.")
                return genai.Client(api_key=api_key)
            else:
                print(f"Using Vertex AI mode for GenAI (Project: {project_id}, Location: {location}).")
                return genai.Client(vertexai=True, project=project_id, location=location)
        except Exception as e:
            print(f"Error initializing GenAI Client: {e}")
            raise ConnectionError(f"GenAI initialization failed: {e}")

    def _initialize_openai(self, api_key: str, base_url: str = None):
        if not openai:
            raise ImportError("openai package is not installed. Please run `pip install openai`")
        
        print(f"Initializing OpenAI Client...")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
             raise ValueError("OpenAI API Key is required. Please provide it via --api_key or OPENAI_API_KEY environment variable.")

        try:
            return openai.OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            print(f"Error initializing OpenAI Client: {e}")
            raise ConnectionError(f"OpenAI initialization failed: {e}")

    def ask(self, inputs: List[Union[str, Path, 'Image.Image']], response_schema: dict = None, verbose: bool = False, max_retries: int = 3, retry_delay: int = 2) -> Union[str, dict]:
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "vertexai":
                    return self._ask_vertexai(inputs, response_schema, verbose)
                elif self.provider == "genai":
                    return self._ask_genai(inputs, response_schema, verbose)
                elif self.provider == "openai":
                    return self._ask_openai(inputs, response_schema, verbose)
                return ""
            except RetryableError as e:
                last_exception = e
                print(f"Attempt {attempt + 1}/{max_retries + 1} failed (retryable): {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay * 2**attempt)
            except Exception as e:
                print(f"Non-retryable error occurred: {e}")
                raise
        
        raise last_exception

    def _ask_vertexai(self, inputs, response_schema, verbose):
        if verbose:
            print("\nProcessing inputs (VertexAI Legacy)...")
        contents = []
        for item in inputs:
            if Image and isinstance(item, Image.Image):
                if verbose:
                    print("  - Adding PIL image object")
                buffer = io.BytesIO()
                item.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                contents.append(Part.from_data(data=image_bytes, mime_type="image/png"))
            elif isinstance(item, str):
                if verbose:
                    print(f"  - Adding text: \"{item}\"")
                contents.append(item)
            else:
                if verbose:
                    print(f"  - Warning: Skipping unsupported input type: {type(item)}")

        if not contents:
            raise ValueError("Input list is empty or contains no valid items.")

        generation_config = GenerationConfig(max_output_tokens=4096)
        if response_schema:
            if verbose:
                print("  - Applying JSON response schema.")
            generation_config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=4096,
            )

        if verbose:
            print("\nGenerating answer...")
        try:
            response = self.model.generate_content(contents, generation_config=generation_config)
            
            if response.candidates and response.candidates[0].content.parts:
                raw_text = response.candidates[0].content.parts[0].text
                if response_schema:
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON response: {e}")
                        print(f"Raw response from model: {raw_text}")
                        raise RetryableError(f"Failed to parse JSON response: {e}")
                else:
                    return raw_text
            else:
                raise RetryableError(f"No answer generated. Full response: {response}")
        except RetryableError:
            raise
        except Exception as e:
            print(f"An error occurred during content generation: {e}")
            raise RetryableError(f"Content generation failed: {e}")

    def _ask_genai(self, inputs, response_schema, verbose):
        if verbose:
            print("\nProcessing inputs (GenAI)...")
        
        contents = []
        for item in inputs:
            if isinstance(item, str):
                 contents.append(item)
            elif Image and isinstance(item, Image.Image):
                # GenAI SDK supports PIL Image directly
                if verbose:
                    print("  - Adding PIL image object")
                contents.append(item)
            else:
                if verbose:
                    print(f"  - Warning: Skipping or passing through input type: {type(item)}")
                contents.append(item)

        config_args = {'max_output_tokens': 4096}
        if response_schema:
             config_args['response_mime_type'] = 'application/json'
             config_args['response_schema'] = response_schema
        
        config = genai_types.GenerateContentConfig(**config_args)

        if verbose:
            print(f"\nGenerating answer with model {self.model_name}...")
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            if response.text:
                raw_text = response.text
                if response_schema:
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        # Try to find JSON block if text contains markdown
                        import re
                        match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(1))
                            except json.JSONDecodeError as e:
                                raise RetryableError(f"Failed to parse JSON from markdown block: {e}")
                        match = re.search(r'```\n(.*?)\n```', raw_text, re.DOTALL)
                        if match:
                             # Sometimes it just returns ``` ... ```
                            try:
                                return json.loads(match.group(1))
                            except json.JSONDecodeError as e:
                                raise RetryableError(f"Failed to parse JSON from markdown block: {e}")
                        # If no markdown block found, try parsing raw text
                        try:
                            return json.loads(raw_text)
                        except json.JSONDecodeError as e:
                            raise RetryableError(f"Failed to parse JSON response: {e}")
                return raw_text
            else:
                print(f"Full response: {response}")
                raise RetryableError("No text in response.")
                
        except RetryableError:
            raise
        except Exception as e:
            print(f"GenAI generation error: {e}")
            raise RetryableError(f"GenAI generation failed: {e}")

    def _ask_openai(self, inputs, response_schema, verbose):
        if verbose:
            print("\nProcessing inputs (OpenAI)...")
        
        messages = [{"role": "user", "content": []}]
        
        for item in inputs:
            if isinstance(item, str):
                messages[0]["content"].append({"type": "text", "text": item})
            elif Image and isinstance(item, Image.Image):
                # Convert PIL Image to base64
                if verbose:
                    print("  - Adding PIL image object (converting to base64)")
                buffer = io.BytesIO()
                item.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4096
        }
        
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
            # Explicitly ask for JSON in the prompt to ensure compliance for some models
            messages[0]["content"].append({
                "type": "text", 
                "text": f"\nPlease output valid JSON adhering to this schema: {json.dumps(response_schema)}"
            })

        if verbose:
            print(f"\nGenerating answer with OpenAI model {self.model_name}...")

        try:
            completion = self.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            
            if response_schema:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response from model: {content}")
                    raise RetryableError(f"Failed to parse JSON response: {e}")
            return content
        except RetryableError:
            raise
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            raise RetryableError(f"OpenAI generation failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Ask a question about one or more images and text prompts using Gemini or OpenAI.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", 
        required=True, 
        nargs='+', 
        help="A list of inputs, which can be image paths or text prompts.\n"
             "Example: --input image1.jpg 'What is this?' image2.jpg 'Compare it with this one.'"
    )
    parser.add_argument(
        "--schema",
        help="Path to a JSON file defining the response schema. If provided, the output will be JSON."
    )
    
    # New arguments for provider and model configuration
    parser.add_argument(
        "--provider",
        choices=["vertexai", "genai", "openai"],
        default="vertexai",
        help="Backend provider to use. Defaults to 'vertexai' (legacy)."
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name to use."
    )
    parser.add_argument(
        "--api_key",
        help="API Key for OpenAI or Gemini (if not using vertexai credentials)."
    )
    parser.add_argument(
        "--base_url",
        help="Base URL for OpenAI compatible endpoints."
    )
    parser.add_argument(
        "--project_id",
        default="tencent-gemini-omd01",
        help="Project ID for Google Cloud."
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Location for Google Cloud."
    )
    parser.add_argument(
        "--credential_file",
        default="/apdcephfs_sh2/share_300000800/user/leike/interns/tianyuan/research/reason_gen_code/reason_gen/eval/gemini_api.json",
        help="Credential file for Google Cloud."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries for API calls."
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=2,
        help="Delay between retries in seconds."
    )
    args = parser.parse_args()

    schema = None
    if args.schema:
        print(f"Loading response schema from: {args.schema}")
        try:
            with open(args.schema, 'r', encoding='utf-8') as f:
                schema = json.load(f)
        except FileNotFoundError:
            print(f"Error: Schema file not found at {args.schema}")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON from schema file '{args.schema}': {e}")
            return

    try:
        multimodal_qa = GeminiImageQA(
            project_id=args.project_id, 
            location=args.location, 
            credential_file=args.credential_file,
            model_name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        
        inputs = []
        for item in args.input:
            if os.path.exists(item) and mimetypes.guess_type(item)[0] and mimetypes.guess_type(item)[0].startswith('image'):
                if Image:
                    try:
                        inputs.append(Image.open(item))
                    except Exception as e:
                         print(f"Error opening image {item}: {e}. Treating as text.")
                         inputs.append(item)
                else:
                     print(f"Pillow not installed, cannot process image {item}. Treating as text.")
                     inputs.append(item)
            else:
                inputs.append(item)

        answer = multimodal_qa.ask(
            inputs=inputs, 
            response_schema=schema, 
            verbose=True,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        print("\n--- Answer ---")
        if isinstance(answer, dict):
            print(json.dumps(answer, indent=2, ensure_ascii=False))
        else:
            print(answer)
        print("--------------")

    except (ValueError, FileNotFoundError, IOError, ConnectionError, RuntimeError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
