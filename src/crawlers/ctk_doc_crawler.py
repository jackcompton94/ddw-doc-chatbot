from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from src import util
import json
import os


def extract_data_from_page(driver, json_file_path):
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    title = soup.find('h1').text
    content_div = soup.find('div', class_='markdown-body ng-non-bindable')
    content = content_div.get_text(strip=True)
    url = driver.current_url

    page_data = {
        "title": title,
        "content": content,
        "url": url
    }
    util.add_page(page_data, json_file_path)
    return title, content, url, soup


def crawl_pages(start_url, json_file_path):
    stop_condition = 'Onboarding Introduction'
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    try:
        current_url = start_url

        while True:
            driver.get(current_url)

            # Wait for the "Next Page" link to be clickable
            next_page_link = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'nav[aria-label="Pagination"] a[aria-label^="Next Page"]'))
            )

            title, content, url, soup = extract_data_from_page(driver, json_file_path)

            print(f"Title: {title}\nContent: {content}\nURL: {url}\n")

            # Check if the "Pagination-link_right" class exists in the nav element
            nav = soup.select_one('nav[aria-label="Pagination"]')
            if nav:
                next_page_link = nav.find('a', attrs={'aria-label': lambda value: value and value.startswith('Next Page')})

                if next_page_link:
                    next_page_href = next_page_link['href']
                    next_page_url = urljoin('https://implementation.data.world', next_page_href)
                    current_url = next_page_url
                else:
                    print("Next page link not found. Exiting.")
                    break
            else:
                print("Navigation element not found. Exiting.")
                break

            # Check if the stop condition is found in the "Next Page" link's aria-label
            if stop_condition in next_page_link.get('aria-label', ''):
                print(f"Found stop condition '{stop_condition}'. Exiting.")
                break

    finally:
        driver.quit()


def scrape_doc_page(ctk_doc_url, json_file_path, embeddings_csv_path):
    crawl_pages(ctk_doc_url, json_file_path)
    util.update_embeddings(json_file_path, embeddings_csv_path)
