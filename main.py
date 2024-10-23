# -*- coding: utf-8 -*-
"""Prod LLM Linkedin Post"""

"""#Imports"""

import feedparser
import requests
from bs4 import BeautifulSoup
import logging
import urllib.parse
import json
from datetime import datetime
from typing import List, Dict, Any
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lxml.html
import lxml.html.clean
from openai import OpenAI
import os
import random
"""#Config"""

# Get the deployment type from environment variables, defaulting to 'dev' if not set
deployment_type = os.getenv("DEPLOYMENT_TYPE", "dev").lower()

# Define a function to fetch environment variables based on deployment type
def get_env_var(var_name_base, deployment_type):
    """
    Retrieves the environment variable value for the given base name and deployment type.

    Args:
        var_name_base (str): The base name of the environment variable.
        deployment_type (str): The deployment type ('prod' or 'dev').

    Returns:
        str: The value of the environment variable.
    """
    var_name = f"{var_name_base}_{deployment_type.upper()}"
    return os.getenv(var_name)

# Fetch LinkedIn credentials based on deployment type
client_id = get_env_var("LINKEDIN_CLIENT_ID", deployment_type)
client_secret = get_env_var("LINKEDIN_CLIENT_SECRET", deployment_type)
auth_token = get_env_var("LINKEDIN_AUTH_TOKEN", deployment_type)
# Set other configurations
redirect_uri = "http://localhost:8000"
visibility = "PUBLIC"  # Options: 'PUBLIC' or 'CONNECTIONS'

# OpenAI credentials
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SIMILARITY_THRESHOLD = 0.03  # Adjust as needed
# Assign weights (preference to publish date)
weight_date = 0.7
weight_similarity = 0.3
KEYWORD_COUNT = 10
HASHTAG_COUNT = 7
no_of_articles_to_publish = 2
# Load NLP models globally
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textrank")

feeds = [
    'https://towardsdatascience.com/feed',
    'https://www.howtogeek.com/feed',
    'https://www.wired.com/feed/rss',
    # 'https://www.theverge.com/rss/index.xml',
    # 'https://www.technologyreview.com/feed/',
    # 'http://news.mit.edu/rss/topic/artificial-intelligence2',
    # 'https://deepmind.com/blog/feed/basic/',
    # 'https://www.unite.ai/feed/',
    # 'https://ai2people.com/feed/',
    # 'https://topmarketingai.com/feed/',
    # 'https://hanhdbrown.com/feed/',
    # 'https://aiparabellum.com/feed/',
    # 'https://www.aihubb.se/rss/2S6wGfhOoJdr/posts',
    # 'https://dailyai.com/feed/',
    # 'https://rss2.feedspot.com/https://vodex.ai/blog.html',
    # 'https://nyheter.aitool.se/feed/',
    # 'https://www.spritle.com/blog/feed/',
    # 'https://yatter.in/feed/',
    # 'https://www.shaip.com/feed/',
    # 'https://zerothprinciples.substack.com/feed',
    # 'https://appzoon.com/feed/',
    # 'https://rss2.feedspot.com/https://automationagencyindia.com/all-blogs',
    # 'https://rss2.feedspot.com/https://visionify.ai/blog/',
    # 'https://airevolution.blog/feed/',
    # 'https://rss2.feedspot.com/https://denser.ai/blog/',
    # 'https://saal.ai/feed/',
    # 'https://aicolabs.info/feed/',
    # 'https://rss2.feedspot.com/https://mrrama.com/',
    # 'https://medium.com/feed/@securechainai',
    # 'https://advancewithai.net/feed',
    # 'https://qudata.com/en/news/rss.xml',
    # 'https://hanhdbrown.com/category/ai/feed/',
    # 'https://www.oreilly.com/radar/topics/ai-ml/feed/index.xml',
    # 'https://blogs.sas.com/content/topic/artificial-intelligence/feed/',
    # 'https://blogs.rstudio.com/ai/index.xml',
    # 'https://www.technologyreview.com/topic/artificial-intelligence/feed',
    # 'https://nanonets.com/blog/rss/',
    # 'https://www.datarobot.com/blog/feed/',
    # 'https://bigdataanalyticsnews.com/category/artificial-intelligence/feed/',
]

"""#FeedManager"""

class FeedManager:
    def __init__(self, feeds: List[str]):
        self.feeds = feeds

    def filter_dead_feeds(self) -> List[str]:
        filtered_feeds = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                if feed.entries:
                    filtered_feeds.append(feed_url)
                    logger.info(f"Feed is active: {feed_url}")
                else:
                    logger.warning(f"No entries found in feed: {feed_url}")
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
        return filtered_feeds

"""#Article"""

class Article:
    def __init__(self, title: str, link: str, published: datetime, first_image_url: str,
                 useful_links: List[str], article_text: str):
        self.title = title
        self.link = link
        self.published = published
        self.first_image_url = first_image_url
        self.useful_links = useful_links
        self.article_text = article_text
        self.keywords = []
        self.hashtags = []
        self.similarity_score = 0.0
        self.date_score = 0.0
        self.combined_score = 0.0
        self.content = ""
    def to_dict(self):
        return {
            'title': self.title,
            'link': self.link,
            'published': self.published.isoformat(),  # Convert datetime to string
            'first_image_url': self.first_image_url,
            'keywords': self.keywords,
            'hashtags': self.hashtags,
            'similarity_score': self.similarity_score,
            'date_score': self.date_score,
            'combined_score': self.combined_score,
            'content': self.content,
        }

    @classmethod
    def from_dict(cls, data):
        article = cls(
            title=data.get('title', 'Untitled'),
            link=data.get('link', ''),
            published=datetime.fromisoformat(data['published']),
            first_image_url=data.get('first_image_url', ''),  # This if it's missing
            useful_links=[],  # Set to an empty list since you're not reading it
            article_text='',  # Set to an empty string since you're not reading it
        )
        article.keywords = data.get('keywords', [])
        article.hashtags = data.get('hashtags', [])
        article.similarity_score = data.get('similarity_score', 0.0)
        article.date_score = data.get('date_score', 0.0)
        article.combined_score = data.get('combined_score', 0.0)
        article.content = data.get('content', "")
        return article



"""#ArticleFetcher"""

# Initialize the logger
logger = logging.getLogger(__name__)

class ArticleFetcher:
    def fetch_articles(self, feed_urls: List[str]) -> List[Article]:
        articles = []
        for feed_url in feed_urls:
            logger.info(f"Fetching articles from feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                try:
                    title = entry.title
                    link = entry.link
                    published_parsed = entry.get('published_parsed')
                    published = datetime(*published_parsed[:6]) if published_parsed else datetime.now()

                    # Fetch the article content
                    response = requests.get(link, timeout=10)
                    response.raise_for_status()
                    html_content = response.text

                    # Parse the HTML content using lxml
                    tree = lxml.html.fromstring(html_content)

                    # Clean the HTML content
                    cleaner = lxml.html.clean.Cleaner(
                        style=True,
                        scripts=True,
                        comments=True,
                        javascript=True,
                        page_structure=False,
                        safe_attrs_only=False
                    )
                    tree = cleaner.clean_html(tree)

                    # Try to extract the main content
                    article_text = ''
                    # Attempt to find <article> tags
                    article_elements = tree.xpath('//article')
                    if article_elements:
                        # Extract text from the first <article> element
                        article_text = article_elements[0].text_content()
                    else:
                        # Fallback: extract text from the body
                        body_elements = tree.xpath('//body')
                        if body_elements:
                            article_text = body_elements[0].text_content()
                        else:
                            logger.warning(f"No <article> or <body> tag found in {link}")
                            continue  # Skip this article

                    # Remove extra whitespace
                    article_text = ' '.join(article_text.split())

                    # Extract useful links
                    useful_links = tree.xpath('//a[@href]')
                    useful_links = [a.get('href') for a in useful_links if a.get('href').startswith('http')]

                    # Extract first image
                    first_image_url = self.extract_first_image(html_content, link)
                    if first_image_url != '':
                        article = Article(
                            title=title,
                            link=link,
                            published=published,
                            first_image_url=first_image_url,
                            useful_links=useful_links,
                            article_text=article_text
                        )
                        articles.append(article)
                        logger.info(f"Article fetched: {title}")
                    else:
                        logger.warning(f"No image found for article: {title}")
                except Exception as e:
                    logger.error(f"Error fetching article from {feed_url}: {e}")
        return articles

    def extract_first_image(self, html_content: str, base_url: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Try to extract og:image
        og_image_tag = soup.find('meta', property='og:image')
        if og_image_tag and og_image_tag.get('content'):
            first_image = og_image_tag.get('content')
            if not first_image.startswith('http'):
                first_image = requests.compat.urljoin(base_url, first_image)
            return first_image
        else:
            # Fallback to first image in the article
            images = soup.find_all('img')
            for img in images:
                img_src = img.get('src') or img.get('data-src')
                if img_src:
                    if not img_src.startswith('http'):
                        img_src = requests.compat.urljoin(base_url, img_src)
                    return img_src
        return ''
"""#ArticlePublishManager"""
class PublishedArticlesManager:
    def __init__(self, json_file='published_articles.json'):
        self.json_file = json_file
        self.published_articles = []
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                try:
                    articles_data = json.load(f)
                    self.published_articles = [Article.from_dict(data) for data in articles_data]
                except json.JSONDecodeError:
                    self.published_articles = []
        else:
            self.published_articles = []

    def is_published(self, article):
        return any(published_article.link == article.link for published_article in self.published_articles)

    def add_published(self, article):
        self.published_articles.append(article)

    def save(self):
        with open(self.json_file, 'w') as f:
            json.dump([article.to_dict() for article in self.published_articles], f, indent=4)

"""#ArticleFilter"""

class ArticleFilter:
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD, keyword_count: int = KEYWORD_COUNT):
        self.similarity_threshold = similarity_threshold
        self.base_keywords = [
            'artificial intelligence', 'machine learning', 'deep learning', 'tech innovation', 'blockchain',
            'quantum computing', 'reinforcement learning', 'technology', 'natural language processing',
            'computer vision', 'big data', 'internet of things', 'cloud computing', 'edge computing',
            '5G technology', 'virtual reality', 'augmented reality', 'cybersecurity', 'data privacy',
            'robotics', 'autonomous vehicles', 'self-driving cars', 'fintech', 'cryptocurrency',
            'decentralized finance', 'NFTs', 'genomics', 'biotechnology', 'digital transformation',
            'digital twin', 'smart cities', 'sustainable technology', 'green energy', 'renewable energy',
            'electric vehicles', 'healthtech', 'medtech', 'wearable technology', 'AI ethics',
            'quantum supremacy', 'neural networks', 'predictive analytics', 'data science',
            'virtual assistants', 'chatbots', 'automation', 'digital economy', 'edge AI',
            'AI in healthcare', 'smart contracts', 'climate tech', 'privacy-enhancing technologies'
        ]
        # Prepare keyword vector
        self.keyword_vector = ' '.join(self.base_keywords)
        self.keyword_count = keyword_count

    def extract_keywords(self, text: str) -> List[str]:
        doc = nlp(text)
        keywords = [phrase.text for phrase in doc._.phrases[:self.keyword_count]]
        logger.info(f"Keywords extracted: {keywords}")
        return keywords

    def generate_hashtags(self, keywords: List[str]) -> List[str]:
        hashtags = ['#' + keyword.replace(' ', '') for keyword in keywords if len(keyword.split()) <= 3]
        logger.info(f"Hashtags generated: {hashtags[:HASHTAG_COUNT]}")
        return hashtags[:HASHTAG_COUNT]

    def filter_articles(self, articles: List[Article]) -> List[Article]:
        filtered_articles = []
        for article in articles:
            content = article.title + ' ' + article.article_text
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([content, self.keyword_vector])
            similarity_score = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
            article.similarity_score = similarity_score
            logger.info(f"Article '{article.title}' similarity score: {similarity_score}")
            filtered_articles.append(article)

        # Compute date_score for each article
        # Get the earliest and latest publish dates among the filtered articles
        publish_dates = [article.published for article in filtered_articles]
        earliest_date = min(publish_dates)
        latest_date = max(publish_dates)
        date_range = (latest_date - earliest_date).total_seconds()
        if date_range == 0:
            date_range = 1  # Avoid division by zero if all dates are the same

        # Compute date_score for each article
        for article in filtered_articles:
            time_diff = (article.published - earliest_date).total_seconds()
            date_score = time_diff / date_range  # Normalize between 0 and 1
            article.date_score = date_score

        for article in filtered_articles:
            # Ensure similarity_score is greater than zero
            similarity = max(article.similarity_score, 1e-6)
            # Compute combined_score as weighted product
            article.combined_score = (article.date_score ** weight_date) * (similarity ** weight_similarity)

        # Sort articles by combined_score descending
        filtered_articles.sort(key=lambda x: x.combined_score, reverse=True)

        # Select the top 5 articles
        filtered_articles = filtered_articles[:no_of_articles_to_publish]

        # Add hashtags
        for article in filtered_articles:
            # Extract keywords and hashtags
            keywords = self.extract_keywords(article.article_text)
            article.keywords = keywords

            hashtags = self.generate_hashtags(keywords)
            article.hashtags = hashtags
        # If 5 or fewer articles passed the threshold, no further action is needed
        return filtered_articles

"""#ContentGenerator"""

from typing import List
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key)

    def generate_summary(self, text: str) -> str:
        """
        Generates a detailed summary of the given text using an AI language model.

        Args:
            text (str): The text of the article to summarize.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in AI and technology, skilled at distilling complex articles into insightful summaries.",
                    },
                    {
                        "role": "user",
                        "content": f"""
Please provide a detailed summary of the following article, focusing on the key innovations, implications, and any notable data or quotes:

{text}

The summary should be 3-4 paragraphs long and capture the essence of the article.
Note: Keep any relevant links unedited and list them at the end with some context.
""",
                    },
                ],
                max_tokens=550,
                temperature=0.7,
            )
            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

    def generate_engaging_content(self, article: Article) -> str:
        """
        Generates engaging content for a LinkedIn post based on the article.

        Args:
            article (Article): The article object containing necessary details.

        Returns:
            str: The generated engaging content.
        """
        summary = self.generate_summary(article.article_text)
        credit = article.link
        hashtags = article.hashtags  # Assuming 'hashtags' is a list of strings

        try:
            prompt_templates = []
            try:
                with open('article_templates.json', 'r', encoding='utf-8') as file:
                    prompt_templates = json.load(file)
            except Exception as e:
                logger.error(f"Error fetching article_templates.json: {e}")
            prompt = random.choice(prompt_templates).replace("{summary}", summary)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional content writer specializing in creating engaging and insightful LinkedIn posts on AI and technology.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            content_body = response.choices[0].message.content.strip()
            logger.info("Engaging content generated")

            # Statistically append the closing remarks
            closing_remarks = f"""

{' '.join(hashtags)}

Your thoughts? 👇

---------

Follow me for more updates on everything AI and Tech

Credit: {credit}
"""
            full_content = f"{content_body}\n{closing_remarks}"
            return full_content
        except Exception as e:
            logger.error(f"Error generating engaging content: {e}")
            return ""

"""#linkedinPoster"""

class LinkedInPoster:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }
        self.person_urn = self.get_linkedin_urn(self.access_token)

    def get_linkedin_urn(self,access_token):

        response = requests.get('https://api.linkedin.com/v2/userinfo', headers=self.headers)
        if response.status_code != 200:
            print(f"Failed to retrieve user profile. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        response_data = response.json()
        linkedin_id = response_data.get('sub')
        if not linkedin_id:
            print("User ID not found in the response.")
            print(f"Response Data: {response_data}")
            return None
        return f'urn:li:person:{linkedin_id}'

    def upload_image(self, image_url: str) -> str:
        """
        Uploads an image from a URL to LinkedIn and returns the asset URN.

        Args:
            image_url (str): The URL of the image to upload.

        Returns:
            str: The asset URN of the uploaded image.
        """
        self.random_number = random.random()
        # Step 1: Download the image from the given URL
        response = requests.get(image_url)
        if response.status_code != 200:
            logger.error("Failed to download image")
            return None
        image_data = response.content

        # Step 2: Register the upload with LinkedIn
        register_upload_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
        register_upload_body = {
            "registerUploadRequest": {
                "recipes": [
                    "urn:li:digitalmediaRecipe:feedshare-image"
                ],
                "owner": self.person_urn,
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }
                ]
            }
        }
        response = requests.post(register_upload_url, headers=self.headers, json=register_upload_body)
        if response.status_code not in (200, 201):
            logger.error("Failed to register upload")
            logger.error(f"Status Code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
        response_data = response.json()
        upload_mechanism = response_data['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']
        upload_url = upload_mechanism['uploadUrl']
        asset = response_data['value']['asset']

        # Step 3: Upload the image binary to LinkedIn
        upload_headers = upload_mechanism.get('headers', {})
        upload_headers["Authorization"] = f"Bearer {self.access_token}"
        upload_headers["Content-Type"] = "application/octet-stream"

        response = requests.put(upload_url, headers=upload_headers, data=image_data)
        if response.status_code not in (200, 201):
            logger.error("Failed to upload image")
            logger.error(f"Status Code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
        logger.info(f"Image uploaded successfully. Asset: {asset}")
        return asset

    def post_content(self, content: str, image_asset: str = None):
        """
        Posts content to LinkedIn, optionally with an image.

        Args:
            content (str): The text content to post.
            image_asset (str, optional): The asset URN of the uploaded image. Defaults to None.
        """
        share_url = "https://api.linkedin.com/v2/ugcPosts"
        share_body = {
            "author": self.person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content,

                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": visibility
            }
        }

        if image_asset:
            share_body["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "IMAGE"
            share_body["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [
                {
                    "status": "READY",
                    "description": {
                        "text": f"Image description {self.random_number}"
                    },
                    "media": image_asset,
                    "title": {
                        "text": f"Image title {self.random_number}"
                    }
                }
            ]

        response = requests.post(share_url, headers=self.headers, json=share_body)
        if response.status_code in (200, 201):
            logger.info("Content posted to LinkedIn successfully")
            logger.info(f"Post URN: {response.headers.get('X-RestLi-Id')}")
        else:
            logger.error("Failed to post content to LinkedIn")
            logger.error(f"Status Code: {response.status_code}")
            logger.error(f"Response: {response.text}")

    def fetch_articles_and_post(self, article: Article):
        """
        Fetches articles and posts them to LinkedIn.

        Args:
            articles (List[Article]): A list of Article objects to post.
        """
        image_asset = self.upload_image(article.first_image_url)
        self.post_content(article.content, image_asset=image_asset)
            # Generate engaging cont

"""#Main"""

# Initialize managers
feed_manager = FeedManager(feeds)
filtered_feeds = feed_manager.filter_dead_feeds()
article_fetcher = ArticleFetcher()

# Initialize the PublishedArticlesManager
published_articles_manager = PublishedArticlesManager()

# Fetch articles
articles = article_fetcher.fetch_articles(filtered_feeds)

# Filter out already published articles
unpublished_articles = [article for article in articles if not published_articles_manager.is_published(article)]

article_filter = ArticleFilter(similarity_threshold=SIMILARITY_THRESHOLD)
filtered_articles = article_filter.filter_articles(unpublished_articles)
content_generator = ContentGenerator()
linkedin_poster = LinkedInPoster(auth_token)
for article in filtered_articles:
    content = content_generator.generate_engaging_content(article)
    article.content = content.replace('**', '').replace('"', '')
    print("------------------------------------------------------------")
    print(article.content)
    linkedin_poster.fetch_articles_and_post(article)

    published_articles_manager.add_published(article)
    published_articles_manager.save()

