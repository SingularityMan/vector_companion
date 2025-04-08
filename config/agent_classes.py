from __future__ import annotations
import os
import asyncio
import time
import random
import requests
import urllib.parse
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin, urlunparse
import re
from bs4 import BeautifulSoup
import subprocess
import base64
from io import BytesIO
import uuid
from pathlib import Path
import mimetypes

from PIL import Image
from duckduckgo_search import DDGS
import praw
import ollama
from config.text_processing import clean_text, split_buffer_into_sentences

class Agent():

    """
    Class of an agent that will be speaking.

    The class is responsible for generating a text response and summarizing the chat history when appropriate.
    """

    def __init__(self, agent_name, agent_gender, personality_traits, system_prompt1, system_prompt2, agent_active, language_model, vision_model, context_length, speaker_wav, extraversion):
        self.agent_name = agent_name
        self.agent_gender = agent_gender
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.previous_agent_message = ""
        self.personality_traits = personality_traits
        self.trait_set = []
        self.agent_active = agent_active
        self.language_model = language_model
        self.vision_model = vision_model
        self.context_length = context_length
        self.speaker_wav = speaker_wav
        self.extraversion = extraversion

    def gather_agent_traits(self, agent_traits: list) -> str:

        """
        Gather the agent traits.
        Returns a string in the form of a concatenated list.
        """
        
        return ''.join(agent_traits)

    async def generate_text_stream(
    self,
    messages: list,
    system_prompt: str,
    user_input: str,
    context_length: int = 32000,
    temperature: float = 0.7,
    top_p: float = 0.3,
    top_k: int = 10000
    ) -> Tuple[List, List, AsyncGenerator[str, None]]:
        """
        Generates a text response as a stream using ollama.chat.
        Returns an asynchronous generator yielding sentences.
        """
        try:

            messages[0] = {"role": "system", "content": system_prompt}
            messages.append({"role": "user", "content": user_input})

            async def fetch_stream():
                loop = asyncio.get_event_loop()
                buffer = ''
                complete_response = ''  # To hold the entire concatenated output
                
                def run_chat():
                    return ollama.chat(
                        model=self.language_model,
                        messages=messages,
                        stream=True,
                        keep_alive=-1,
                        options={
                            "repeat_penalty": 1.15,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "num_ctx": context_length,
                            "num_batch": 128
                        }
                    )

                # Run ollama.chat in the executor to prevent blocking the event loop
                stream = await loop.run_in_executor(None, run_chat)

                for chunk in stream:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        buffer += content
                        sentences, buffer = split_buffer_into_sentences(buffer)
                        for sentence in sentences:
                            # Clean up the sentence
                            sentence = clean_text(sentence)
                            complete_response += sentence + ' '  # Concatenate to the complete response
                            yield sentence

                # Handle any remaining buffer
                if buffer.strip():
                    buffer = clean_text(buffer)
                    complete_response += buffer + ' '  # Add remaining buffer to the complete response

                # Append the complete response as a single message
                complete_response = complete_response.strip()
                messages.append({"role": "assistant", "content": complete_response})
        except Exception as e:
            print("Error generating text: ", e)

        return messages, fetch_stream()
        
class VectorAgent():

    """
    Class of a special agent used to handle agentic tasks.
    Also has the ability to perform simple search and deep search functionality.
    """

    def __init__(self, language_model, analysis_model, vision_model, context_length, audio_model, audio_model_name):
        self.language_model = language_model
        self.analysis_model = analysis_model
        self.vision_model = vision_model
        self.context_length = context_length
        self.audio_model = audio_model
        self.audio_model_name = audio_model_name
        
        # ProxyScrape API URL
        self.api_url = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text"
        try:
            self.proxies = self.load_proxies(self.api_url)
        except:
            self.proxies = None
            pass
        

    def generate_text(self, model, messages, **kwargs):

        while True:
            try:

                kwargs["num_batch"] = 128

                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options=kwargs
                )
                
                text_response = response.get("message", {}).get("content", "No response received")
                if "</think>" in text_response:
                    split_text_response = text_response.split("</think>")
                    text_response = split_text_response[1]
                return text_response.strip()
            except ollama._types.ResponseError as e:
                error_message = str(e)
                print("Error Generating Text: ", error_message)
                if error_message.lower().strip() == "failed to create new sequence: failed to process inputs: image: unknown format":
                    return ""
                kwargs["num_batch"] /= 2
                kwargs["num_ctx"] -= 500
                print("Lowering batch size to", kwargs["num_batch"])
                print("Lowering context length to", kwargs["num_ctx"])
                if kwargs["num_ctx"] <= 2048:
                    return ""
                continue
            except ollama._types.RequestError as e:
                print("Error Generating Text: ", e)
                return ""
            except:
                return ""

    def summarize_text(self, language_model, prompt, **kwargs):
        words = prompt
        print("[ORIGINAL PROMPT]:", words)
        while len(words.split()) > kwargs["num_ctx"]:
            print("Words length: ", len(words.split()))
            chunks = ["".join(words[i:i+kwargs["num_ctx"]]) for i in range(0, len(words.split()), kwargs["num_ctx"])]
            summarized_chunks = []
            for chunk in chunks:
                summary_prompt = f"""
                                 Provide a short and concise summary of the following text:\n\n{chunk}\n\n.
                                 Do not provide any commentary about the text content.
                                 \nIf there is already a summary, do not comment on the summary, only summarize the core subject.
                                 \nMake sure the summary is shorter than the length of the original text.
                                 """
                chunk_message = [{"role": "user", "content": summary_prompt}]
                # Call generate_text to summarize this chunk.
                summary = self.generate_text(language_model, chunk_message, temperature=0.1, num_ctx=kwargs["num_ctx"], num_batch=512)
                print("[CHUNK SUMMARY LENGTH]:", len(summary))
                print("[CHUNK SUMMARY]:\n\n", summary)
                summarized_chunks.append(summary)
            words = "".join(summarized_chunks)
            print("[SUMMARIZED CHUNKS]:", words)
        
        return words
             
    def process_youtube_videos(
        self,
        videos: list,
        audio_model: Any,
        audio_model_name: str,
        analyze_video_model: str = "",
        messages: list = [],
        context_length: int = 4096):

        """
        Uses an agent to download a youtube video.
        Optionally analyzes the video by transcribing it and reading the transcript.

        Returns a response from the analysis model.
        """
        # Set the destination folder and temp folder
        destination_dir = os.path.join("scraped_data", "videos")
        temp_dir = os.path.join(destination_dir, "temp")
        #os.remove(temp_directory)
        os.makedirs(destination_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        destination_dir_list = sorted(os.listdir(destination_dir))
        bat_path = os.path.join(destination_dir, "videos.bat")
        
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("# Your video urls will be placed here for download. Previous urls will be removed when a new search initiates.")
        with open(bat_path, "a", encoding="utf-8") as f:
            for video in videos:
                f.write("\n"+video["link"])

        # Run yt-dlp via subprocess. Install/update if necessary.
        try:
            print("Checking for yt-dlp updates...")
            # Install/update yt-dlp
            subprocess.run(
                "python -m pip install --upgrade yt-dlp",
                shell=True,  # Use shell=True to run multiple commands in a single line
                check=True
            )
            print(f"yt-dlp has been successfully installed or updated.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while managing yt-dlp: {e}")

        # Construct the yt-dlp command
        command = [
            "yt-dlp",
            "-a", bat_path,  # Use the batch file for URLs
            "-P", f"home:{destination_dir}"  # Specify the home path for final downloads
        ]

        try:
            # Update/install ffmpeg
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
            print(f"FFmpeg Version:\n{result.stdout}")
        except subprocess.CalledProcessError:
            print("FFmpeg is not installed or not configured properly. Installing...")
            try:
                # Run the Scoop installation command if ffmpeg is not available.
                subprocess.run(["scoop", "install", "ffmpeg"], check=True)
                print("ffmpeg has been successfully installed via Scoop!")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while installing ffmpeg: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            print("ERROR: ", e)
        return False

        if analyze_video_model != "":
            videos_analysis = self.transcribe_youtube_videos(
                videos,
                destination_dir,
                temp_dir,
                audio_model,
                audio_model_name,
                analyze_video_model,
                messages=[],
                context_length=4096
                )
            return videos_analysis
        else:
            print("All good!")
            return True
        
    def scrape_youtube_videos(
        self,
        query: str = "cats",
        region: str = "us-en",
        safesearch: str = "off",
        timelimit: Optional[str] = None,
        resolution: Optional[str] = None,
        duration: Optional[str] = None,
        max_results: int = 10,
        analyze_video_model: str = "",
        messages: list = [],
        context_length: int = 4096):

        videos = []

        print("Scraping Youtube Videos...")

        results = DDGS().videos(
        keywords=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        resolution=resolution,
        duration=duration,
        max_results=max_results,
    )
        print("[VIDEO RESULTS:]", results)
        
        for result in results:
            
            videos.append(
                {
                    "channel": result["uploader"],
                    "title": result["title"],
                    "description": result["description"],
                    "link": result["content"]
                }
            )

            print(result)
            print("\n\n")
            
        self.process_youtube_videos(videos, self.audio_model, self.audio_model_name, analyze_video_model="", messages=[], context_length=context_length)
        return True
        
            
    def load_proxies(self, api_url):
        try:
            # Send a request to the ProxyScrape API
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            # Split the response into a list of proxies
            proxies = response.text.splitlines()
            print(f"Loaded {len(proxies)} proxies.")
            return proxies
            proxies_accepted = []
            for proxy in proxies:
                proxy_location_api_url = f"https://ipinfo.io/{proxy}/json"
                try:
                    response = requests.get(proxy_location_api_url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    proxy_dict = {
                        "ip": data.get("ip"),
                        "country": data.get("country"),
                        "region": data.get("region"),
                        "city": data.get("city")
                    }
                    #print("proxy location:", proxy_dict["country"])
                    if proxy_dict["country"].strip() == "US":
                        proxies_accepted.append(proxy)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            return proxies_accepted
        except Exception as e:
            print(f"Failed to load proxies: {e}")
            return None

    def vector_search(
        self,
        language_model,
        vision_model,
        prompt: str,
        messages: list,
        context_length: int,
        search_override: str = "",
        ) -> str:

        """
        Performs a web search
        """
            
        if self.proxies is None:
            try:
                self.proxies = self.load_proxies(self.api_url)
            except:
                self.proxies = None
        self.working_proxy = None
        self.user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0"
    ]

        query_prompt = f"""
        \nYou are a query generator bot.
        \nRead the prompt below to determine if the user specifically requested a search or not.
        \nIf the the user is directly requesting an online search, evaluate the following:
        \n\n- If the user has NOT explicitly requested a deep search but did request an online search, just generate a simple, one-sentence search query to answer the user's inquiry, with quotation marks surrounding the main subject of the query, but not the entire query.
        \n- If the user explicitly requested a deep search, only reply with 'deep search' and nothing more. The user has to explicitly mention the phrase 'deep search'. Otherwise, assume a normal search query.
        \n- If the user has not explicitly requested neither deep search nor online search, reply with a one-word 'no' response.
        \n- Either form of search needs to be specifically requested, not implied, by the user. Having the user simply ask a question is not enough to confirm if a search is required.
        \n\nHere is the user's message:
        \n\n{prompt}\n\n
        \nThe entire output should be generated in any of the applicable structures mentioned above, depending on the user's message, unless explicitly overriden.
        \n\n{search_override}
        Follow all of these instructions without providing any additional commentary at all."""
        
        messages.append({"role": "user", "content": query_prompt})
        search_query = self.generate_text(language_model, messages[-10:], temperature=0.1, num_ctx=context_length, num_batch=512)
        fixed_search_query = search_query.replace('"', '')
        messages.append({"role": "assistant", "content": fixed_search_query})
        #if search_override != "":
            #fixed_search_query = search_override
        print("[Vector Query]:", fixed_search_query)

        if fixed_search_query.lower().strip() in ["no", "no."]:
            return ""
        elif fixed_search_query.strip() == "deep search":
            deep_search_prompt = f"""
            \nYou are a query generator bot with Deep Search mode activated using DuckDuckGo's web search API.
            \n\nSearch online by generating a query in order to anwer the user's original inquiry:"
            \n\n{prompt}\n\n
            \nThe query should be focused on directly answering the user's question in relation to the conversation history as well.
            \nThe structure of your response needs to be a single query. 
            \nVary your attempts as much as possible if previous attempts failed.
            \nFollow all of these instructions without providing any additional commentary or acknowledgement at all.
            \nThe resulting output should be a single search query. Nothing more. """
            
            messages.append({"role": "user", "content": deep_search_prompt})
            deep_search_query = self.generate_text(language_model, messages[-10:], temperature=0.7, num_ctx=context_length, num_batch=512)
            deep_search_query_fixed = deep_search_query.replace('"', "")
            messages.append({"role": "assistant", "content": deep_search_query_fixed})
            print("[Vector Deep Search Query]:", deep_search_query_fixed)

            search_parameters_list = [
                                "latest",
                                "-facebook",
                                "-instagram",
                                "-twitter",
                                "-reddit",
                            ]
            search_parameters = " ".join(search_parameters_list)
            
            search_results = self.get_duckduckgo_text_snippet(
                prompt,
                deep_search_query_fixed +" "+search_parameters,
                language_model,
                messages,
                context_length,
                search_type="[deep search]",
                backend="auto",
                max_results=10,
                working_proxy=self.working_proxy,
                proxies=self.proxies,
                user_agent=random.choice(self.user_agents)
                )
            
            if search_results == ["No results found for your query."] or len(search_results) == 0:
                print("No results found.")
                return ""
                
            search_results = self.summarize_text(language_model, " ".join(search_results), temperature=0.1, num_ctx=context_length, num_batch=512)

            deep_search_eval_prompt = f"""
                     Read the text below and determine if it directly answers the user's inquiry.
                     If so, reply with a one-word 'yes' response.
                     If not, reply with a one-word 'no' response.
                     \nHere is the user's inquiry:
                     \n\n{prompt}\n\n
                     Here is the information gathered online:
                     \n\n{search_results}\n\n
                     The output needs to be a one-word 'yes' or 'no' response.
                     \nThese instructions are strict.
                     \nFollow these instructions without providing any commentary or mentioning them.
                     """
            
            messages.append({"role": "user", "content": deep_search_eval_prompt})
            results_eval = self.generate_text(language_model, messages[-10:], temperature=0.1, num_ctx=context_length, num_batch=512)
            messages.append({"role": "assistant", "content": results_eval})
            print("[Vector search complete?]:", results_eval)
            if results_eval.lower().strip() not in  ["no", "no."]:
                summarized_prompt = f"""{search_results}"""
                #messages.append({"role": "user", "content": summarized_prompt})
                summarized_response = self.summarize_text(language_model, summarized_prompt, temperature=0.1, num_ctx=context_length, num_batch=512)
                #messages.append({"role": "assistant", "content": summarized_response})
                reddit_search_results = self.search_reddit(prompt, deep_search_query_fixed, language_model, vision_model, messages[-10:], context_length, limit=10)
                summarized_response += "\n\nReddit search results:\n\n"+reddit_search_results
                final_response = self.summarize_text(language_model, summarized_response, temperature=0.1, num_ctx=context_length, num_batch=512)
                print("[FINAL RESULT]:", final_response)
                return f"""This is an instruction override in response to the user's deep search request.
                       \nThe following text contains detailed information gathered from an online deep search result:
                       \n\n{final_response.strip()}\n\n
                       Generate a detailed, 10-sentence response based on this information."""
            return "No information available."
        else:
            search_results = self.get_duckduckgo_text_snippet(
                    prompt,
                    fixed_search_query,
                    language_model,
                    messages[-10:],
                    context_length,
                    backend="auto",
                    max_results=15,
                    working_proxy=self.working_proxy,
                    proxies=self.proxies,
                    user_agent=random.choice(self.user_agents)
                    )

            if search_results == ["No results found for your query."] or len(search_results) == 0:
                print("No results found.")
                return ""

            search_results = search_results.strip()
            
            return f"""This is an instruction override.
                   \nThe following text contains detailed information gathered from an online simple search result:
                   \n\n{search_results}\n\n
                   Generate a 4-sentence response based on this information, with the first message containing a maximum of 10 words."""
    
    def is_allowed_to_scrape(self, url: str, user_agent: str = "*") -> bool:
        """
        Checks if scraping the given URL is allowed by the site's robots.txt.
        
        Args:
            url (str): The URL to check.
            user_agent (str): The user agent string (default is "*" to match any agent).
        
        Returns:
            bool: True if allowed, False otherwise.
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as e:
            # If we can't read robots.txt, it's safer to assume scraping is disallowed.
            print(f"Could not read robots.txt from {robots_url}: {e}")
            return False
    
        return rp.can_fetch(url, user_agent)

    def canonicalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        # Force scheme and netloc to lower-case
        return urlunparse((
            parsed.scheme.lower(), 
            parsed.netloc.lower(), 
            parsed.path.rstrip('/'),  # Remove trailing slash for consistency
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

    def download_and_encode_image(self, url: str) -> str:
        # Define the target directory and ensure it exists.
        directory = os.path.join("scraped_data", "images", "temporary_images")
        os.makedirs(directory, exist_ok=True)

        # Get the MIME type from the URL or data
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type and mime_type.startswith("image"):
            pass
        else:
            return None

        if url.endswith(".svg"):
            print(f"Skipping SVG file from {url}.")
            return None

        # Download the image data.
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                image_data = base64.b64decode(encoded)
            except Exception as e:
                print(f"Error decoding data URL: {e}")
                return None
        else:
            try:
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                image_data = response.content
            except Exception as e:
                print(f"Error downloading image from {url}: {e}")
                return None

        # Try to determine the image format.
        try:
            image = Image.open(BytesIO(image_data))
            image.verify()  # Verify it's a valid image.
            fmt = image.format.lower()  # e.g. 'jpeg', 'png'
            extension = ".jpg" if fmt == "jpeg" else f".{fmt}"
        except Exception as e:
            print(f"Downloaded image file from {url} is not a valid image: {e}")
            return None

        # Generate a unique filename.
        filename = f"{uuid.uuid4().hex}{extension}"
        file_path = os.path.join(directory, filename)
        
        # Save the image data to file.
        try:
            with open(file_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            print(f"Error saving image to {file_path}: {e}")
            return None

        # Return an absolute, POSIX-formatted path.
        abs_path = Path(file_path).resolve().as_posix()
        return abs_path


    def extract_links_and_text(self, user_query, language_model, messages, context_length, url, user_agent, visited=None, depth=0, max_depth=2):

        headers = {
        "User-Agent": random.choice(self.user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }

        # Some sites might not support HEAD; you can also use GET with stream=True if needed.
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            pass
        else:
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            if response.status_code != 200:
                return ""
    
        if visited is None:
            visited = set()

        if depth > max_depth:
            return ""
        
        # Normalize URL
        norm_url = self.canonicalize_url(url)
        
        if norm_url in visited:
            return ""
        visited.add(norm_url)
        
        parsed = urlparse(norm_url)
        if parsed.scheme not in ["http", "https"]:
            return ""
        
        try:
            # Fetch page content
            user_agent = random.choice(self.user_agents)
            response = requests.get(norm_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="", strip=True)
        
            links = soup.find_all("a")
            link_list = []
            for link in links:
                try:
                    href = link.get("href")
                    if href:
                        full_url = self.canonicalize_url(urljoin(norm_url, href))
                        link_list.append(full_url)
                        # Optionally, you might immediately add full_url to visited here to prevent duplicate processing.
                except Exception as e:
                    print("Error processing link: ", e, "Continuing")
                    continue

            if len(page_text.split()) > context_length:
                page_text = self.summarize_text(language_model, page_text, temperature=0.1, num_ctx=context_length, num_batch=512)
            
            if link_list:
                joined_links = "\n".join(link_list)
                print(joined_links)
                agent_inquiry_prompt = f"""Read the user's inquiry: \n\n{user_query}\n\n
                                          Here are the text contents extracted from the current webpage:\n\n{page_text}\n\n
                                          Read the conversation history and the page text in order to systematically explore the search results of the inquiry further.
                                          \nYou will generate a list of follow-up questions related to the user's inquiry and the information currently gathered in order to guide your information extraction process.
                                          \nThe purpose of the questioning process is to expand the depth of your search, so feel free to ask deeper questions based on the overall objective.
                                          \nThe entire output must be the list of questions only, nothing more.
                                          \nThese instructions are strict. Absolutely no commentary on the instructions allowed.
                                           """
                messages.append({"role": "user", "content": agent_inquiry_prompt})
                generated_questions = self.generate_text(language_model, messages[-10:], temperature=0.1, num_ctx=context_length, num_batch=512)
                messages.append({"role": "assistant", "content": generated_questions})
                print(generated_questions)
                link_selection_prompt = f"""Read the user's inquiry: \n\n{user_query}\n\n
                                           Here is a list of links:\n\n{joined_links}\n\n
                                           Here is a list of questions you generated to guide your response:\n\n{generated_questions}\n\n
                                           
                                           if any of these links may be related to the user's query, and the list of questions generated in your investigation, respond only with one of these links and nothing else.
                                           Otherwise, reply with a one-word 'no' response.
                                           \n\nThe format of the response can only include either the selected link or a 'no' response, nothing more.
                                           \nThese instructions are strict. Absolutely no commentary on the instructions allowed.
                                           """
                messages.append({"role": "user", "content": link_selection_prompt})
                link_selection = self.generate_text(language_model, messages[-10:], temperature=0.7, num_ctx=context_length, num_batch=512).strip()
                messages.append({"role": "assistant", "content": link_selection})
                if "no" not in link_selection.lower() and not link_selection.lower() in visited:
                    visited.add(link_selection.lower())
                    print("[Potential link found]: ", link_selection, " accessing link...")
                    page_text += "" + self.extract_links_and_text(user_query, language_model, messages, context_length, link_selection, user_agent, visited, depth + 1, max_depth)
                else:
                    print("[No related link found, continuing.]")
        
            return page_text
        
        except Exception as e:
            print(f"Error processing {norm_url}: {e}")
            return ""

    # Helper function to download images
    def download_reddit_image(self, url, filename):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded image: {filename}")
                return True
            else:
                print(f"Failed to download image from {url}")
                return False
        except:
            return False

    def search_reddit(
        self,
        user_query: str,
        query: str,
        language_model: str,
        vision_model: str,
        messages: list,
        context_length: int,
        image_directory: str = "scraped_data\\images",
        limit: int = 5
        ):

        subreddit_query = f"""Read the following information and conversation history to determine which subreddits most likely have the answer the following queries:
                          \n\nUser query: {user_query}\n\n
                          \n\nAI query: {query}\n\n
                          Generate a string of a variable number of subreddits to search in.
                          You can only include the strings by separating each entry with whitespace, not newlines, commas or any other separator.
                          \n\nExample: r/subreddit1 r/subreddit2 r/subreddit3
                          The output must follow the exact format of the example, with additional subreddits included in the string, depending on the queries.
                          Follow these instructions without any commentary at all, these instructions are strict.
                          """

        messages.append({"role": "user", "content": subreddit_query})
        subreddit_query = self.generate_text(language_model, messages[-10:], temperature=0.1, num_ctx=context_length, num_batch=512)
        messages.append({"role": "assistant", "content": subreddit_query})
        if subreddit_query == "":
            return ""
        fixed_subreddit_query = subreddit_query.strip().replace("r/", "")
        fixed_subreddit_query_list = "+".join(fixed_subreddit_query.split())
        print("[SUBREDDIT LIST]: ", fixed_subreddit_query_list)

        # Initialize the Reddit instance
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ["REDDIT_USER_AGENT"]
        )

        all_posts = []
        subreddit = reddit.subreddit(fixed_subreddit_query_list)
        search_results = subreddit.search(query, limit=5, sort="relevance", time_filter="all")
        sorted_search_results = sorted(search_results, key=lambda submission: submission.created_utc, reverse=True)

        for submission in sorted_search_results:
            # Create a dictionary to hold post content.
            post_content = {}
            post_content["title"] = submission.title
            post_content["url"] = submission.url
            print(f"Title: {post_content['title']}")
            print(f"Post URL: {post_content['url']}")

            # Process text or image posts
            post_images = []
            if submission.is_self:
                post_content["body"] = submission.selftext
                # Extract image links embedded in the body
                image_links = re.findall(r'(https?://[^\s]+(?:jpg|png|gif))', post_content["body"])
                for image_url in image_links:
                    image_dir = os.path.join(image_directory, "temp_image.jpg")
                    if self.download_reddit_image(image_url, image_dir):
                        with open(image_dir, 'rb') as img_file:
                            image_data = img_file.read()
                        image_messages = [{"role": "user", "body": "Provide a brief description of this image.", "images": [image_data]}]
                        try:
                            image_review = self.generate_text(vision_model, image_messages, temperature=0.1, num_ctx=512, num_batch=512)
                        except:
                            break
                        post_images.append(image_review)
            else:
                post_content["body"] = "No Body available."
                if hasattr(submission, 'gallery_data'):
                    # Extract image URLs from gallery metadata
                    for item in submission.gallery_data['items'][:5]:
                        media_id = item['media_id']
                        image_url = submission.media_metadata[media_id]['s']['u']
                        image_dir = os.path.join(image_directory, "temp_image.jpg")
                        if self.download_reddit_image(image_url, image_dir):
                            with open(image_dir, 'rb') as img_file:
                                image_data = img_file.read()
                            image_messages = [{"role": "user", "body": "Provide a brief description of this image.", "images": [image_data]}]
                            try:
                                image_review = self.generate_text(vision_model, image_messages, temperature=0.1, num_ctx=512, num_batch=512)
                            except:
                                break
                            post_images.append(image_review)
                elif submission.url.endswith(('.jpg', '.png', '.gif')):
                    # Direct image posts
                    image_dir = os.path.join(image_directory, "temp_image.jpg")
                    if self.download_reddit_image(submission.url, image_dir):
                        with open(image_dir, 'rb') as img_file:
                            image_data = img_file.read()
                        image_messages = [{"role": "user", "content": "Provide a brief description of this image.", "images": [image_data]}]
                        try:
                            image_review = self.generate_text(vision_model, image_messages, temperature=0.1, num_ctx=512, num_batch=512)
                            post_images.append("Image {}: ".format(len(post_images)) + image_review)
                        except:
                            continue

            post_content["images"] = "\n\n".join(post_images)

            # Build post evaluation prompt.
            post_eval_prompt = f"""Review this post's content and determine if it answers the user's and the AI's query:
            
            \n\nUser Query:\n\n{user_query}\n\n

            Here is the AI's query:

            \n\n{query}\n\n

            Here is the post's content:
            \n\nTitle: {post_content['title']}

            \n\nBody: \n\n{post_content['body']}

            \n\nImages (if applicable): \n\n{post_content['images']}

            If the information presented answers the user's and the AI's query, reply with a one-word 'yes' response.
            Otherwise, reply with a one-word 'no' response.
            The content needs to be related to the queries mentioned and the conversation history.
            The response can only be a one-word yes or no response.
            Follow these instructions without any additional commentary or mentioning them. These instructions are strict."""

            messages.append({"role": "user", "content": post_eval_prompt})
            post_review = self.generate_text(language_model, messages[-10:], temperature=0.1, num_ctx=context_length, num_batch=512)
            if post_review.lower().strip() in ("yes", "yes."):
                submission.comments.replace_more(limit=10)
                sorted_comments = sorted(
                submission.comments.list(),
                key=lambda comment: comment.score,
                reverse=True  # Sort in descending order
            )
                all_comments = []
                for comment in sorted_comments[:20]:
                    author = comment.author.name if comment.author else '[Deleted]'
                    all_comments.append(f"Redditor: {author}\n\n{comment.body}")
                post_content["comments"] = self.summarize_text(language_model, "\n\n".join(all_comments), temperature=0.1, num_ctx=context_length, num_batch=512)
                all_posts.append(post_content)
            else:
                print("[IRRELEVANT, CONTINUING.]")
                continue

        # Flatten all posts into one concatenated string.
        def flatten_post(post):
            flat = ""
            for value in post.values():
                if isinstance(value, list):
                    flat += " ".join(value) + " "
                else:
                    flat += str(value) + " "
            return flat.strip()

        all_posts = "\n\n".join(flatten_post(post) for post in all_posts)
        all_posts_summary = self.summarize_text(language_model, all_posts, temperature=0.1, num_ctx=context_length, num_batch=512)
        print("[SUMMARY OF REDDIT RESULTS]: ", all_posts_summary)

        return all_posts_summary
            

    def get_duckduckgo_text_snippet(
        self,
        user_query: str,
        query: str,
        language_model,
        messages: list,
        context_length: int,
        search_type: str = "[normal]",
        region: str = "us-en",
        safesearch: str = "off",
        timelimit: Optional[str] = None,
        backend: str = "auto",
        max_results: Optional[int] = None,
        user_agent: str = "Mozilla/5.0 (compatible; MyResearchBot/1.0)",
        working_proxy: str = None,
        proxies: Optional[list] = None  # Accept a list of proxies
    ) -> str:
        """
        Performs a text search using DuckDuckGo and returns the snippet text from the first result.
        
        Args:
            query (str): The search query.
            messages (list): Conversation history
            context_length (str): LLM context_length
            region (str): Region code (e.g., "wt-wt", "us-en", "uk-en"). Defaults to "wt-wt".
            safesearch (str): Safe search level ("on", "moderate", "off"). Defaults to "moderate".
            timelimit (str | None): Time filter for results ("d", "w", "m", "y"). Defaults to None.
            backend (str): Which backend to use ("auto", "html", "lite"). Defaults to "auto".
            max_results (int | None): Maximum number of results to return. Defaults to first page results.
            user_agent (str): Custom User-Agent string. Defaults to a generic bot User-Agent.
            proxies (list | None): List of proxy strings. Defaults to None.

        Returns:
            str: The snippet text (body) from the first search result,an error message if unavailable or multiple text chunks extracted recursively inside the links associated with the text results.
        """
        attempt_count = 0
        proxy = working_proxy
        non_proxy_attempted = False
        headers = {
        "User-Agent": random.choice(self.user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }
        while True:
            attempt_count = 1
            attempt_time = attempt_count * 1
            user_agent = random.choice(self.user_agents)
            try:
                # Select a random proxy from the list
                if proxies and non_proxy_attempted is True:
                    if working_proxy is None:
                        proxy = random.choice(proxies)
                    ddgs = DDGS(proxy=proxy, headers=headers)
                else:
                    ddgs = DDGS(headers=headers)
                
                # Initialize DDGS instance with proxy support
                results = ddgs.text(
                    keywords=query,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    backend=backend,
                    max_results=max_results
                )
                self.working_proxy = proxy
                working_proxy = self.working_proxy
                break
            except Exception as e:
                print("Search failed:", e, "Retrying..")
                non_proxy_attempted = True
                working_proxy = None
                self.working_proxy = None
                time.sleep(attempt_time)  # Wait before retrying
            
        if not results:
            return ["No results found for your query."]

        # Process and return results
        if search_type != "[deep search]":
            # Extract text snippets (e.g., body field)
            text_results = [snippet.get("body", "") for snippet in results]
            if text_results and any(text_results):
                print("[TEXT SEARCH RESULTS]", text_results)
                return " ".join(text_results)
            else:
                return ["No results found for your query."]
        else:
            # Deep search mode logic (handles URL scraping)
            allowed_results = []
            for result in results:
                url = result.get("href", False)
                if url and self.is_allowed_to_scrape(url, user_agent=user_agent):
                    try:
                        text_result = result.get("body", "No extractable text from result page.") 
                        eval_prompt = f"""
                                      Read this text and reply with a one-word 'yes' response if it is related to the user's inquiry.
                                      \nHere is the user's query:\n\n{user_query}\n\n
                                      \nOtherwise, response with a one-word 'no' response:
                                      {text_result}
                                      \nFollow these instructions without any additional commentary.
                                      """
                        messages.append({"role": "user", "content": eval_prompt})
                        results_eval = self.generate_text(language_model, messages[-5:], temperature=0.1, num_ctx=context_length, num_batch=512)
                        messages.append({"role": "assistant", "content": results_eval})
                        if results_eval.lower().strip() == "yes" or results_eval.lower().strip() == "yes.":
                            user_agent = random.choice(self.user_agents)
                            # Fetch and recursively scrape links
                            text = self.extract_links_and_text(user_query, language_model, messages, context_length, url, user_agent)
                            allowed_results.append(text)
                        else:
                            print("[IRRELEVANT. SKIPPING.]")
                    except Exception as e:
                        print(f"Error fetching or processing {url}: {e}")
                else:
                    text_result = result.get("body", "No extractable text from result page.")
                    if text_result.strip():
                        allowed_results.append(text_result)
                    else:
                        print("[UNABLE TO EXTRACT TEXT]")
            
            return allowed_results
        
