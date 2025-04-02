from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, ElementNotInteractableException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import time
import json
import requests
import os
def create_link_queue(link_file):
    d = dict()
    with open(link_file, "r") as f:
        data = f.read().split("\n")
        for link in data:
            link = "https://" + link
            d[link] = False

    with open("../../data/interim/scraping/has_scraped.json", 'wb') as f:
        f.write(json.dumps(d, indent = 4, ensure_ascii=False).encode("utf8"))



def page_exists(driver):
    try:
        header = driver.find_element(By.TAG_NAME, 'h1')
        if header.text == 'This recipe is exclusively available in the Tasty app.' or header.text == "Oops! We can't find the page you're looking for." or header.text == "Uh-oh. We're experiencing some technical difficulties.":
            return True
    except NoSuchElementException:
            return "NA"
    

def grab_date(driver):
    try:
        date = driver.find_element(By.CLASS_NAME, 'date-field')
        return date.text
    except NoSuchElementException:
        return "NA"
    

def grab_recipe_rating(driver):
    try:
        rating = driver.find_element(By.CLASS_NAME, 'tips-score-heading')
        return rating.text.split("%")[0]
    except NoSuchElementException:
        return "NA"
    

def grab_nutrition(driver):
    try: 
        nutr_info = driver.find_element(By.CLASS_NAME, 'nutrition-details')
        elem = nutr_info.find_element(By.CLASS_NAME,'xs-mt1')
        li = elem.find_elements(By.CLASS_NAME,'list-unstyled')
        nutrition = []
        for j in li:
            html = j.get_attribute('innerHTML')
            html = html.replace('<span class="bold"> <!-- -->',' ')
            html = html.replace('<!-- -->', '')
            html = html.replace('</span>', '')
            nutrition.append(html)
        return nutrition
    except NoSuchElementException:
        return "NA"
    

def grab_comments(driver):
    comments = []
    page = 0
    while True:

        time.sleep(1)

        for wrapper in driver.find_elements(By.ID, 'tip-wrapper'):
            author = wrapper.find_element(By.CLASS_NAME, 'tip-author').text
            text = wrapper.find_element(By.CLASS_NAME, 'tip-body').text
            time_since = wrapper.find_element(By.CLASS_NAME, 'tip-time-since').text
            try:
                rating = wrapper.find_element(By.CLASS_NAME, 'tip-upvotes').text
            except:
                rating = "NA"

            comments.append({'author': author,
                            'text': text,
                            'time_since': time_since,
                            'rating': rating})
            
        page += 1
        print(f'\t\tProcessed comments page: {page}')
        
        try:
            button = driver.find_element(By.CLASS_NAME, 'pagination__button--next')
        except NoSuchElementException:
            break

        if 'disabled' in button.get_attribute('class'):
            break
        
        while True:
            try:    
                button.click()
                break
            except ElementClickInterceptedException:
                actions = ActionChains(driver)
                actions.move_to_element(button).send_keys(Keys.ARROW_DOWN).perform()
                continue
    return comments

def grab_preparation(driver):
    prep_steps = driver.find_element(By.CLASS_NAME, 'prep-steps')
    steps = []
    for step in prep_steps.find_elements(By.CLASS_NAME, 'xs-mb2'):
        steps.append(step.text)
    return steps

def save_and_remove_from_queue(link,dict_to_save):
    #creates the new data file
    json_object = json.dumps({}, indent=4)
    with open("../../data/raw/data_raw.json", "w") as outfile:
        outfile.write(json_object)
    #Saves the newly scaped data
    with open('../../data/raw/data_raw.json', 'r', encoding='utf8') as json_file:
        data_json = json.load(json_file)
    data_json[link] = dict_to_save

    with open("../../data/raw/data_raw.json", 'wb') as f:
        f.write(json.dumps(data_json, indent = 4, ensure_ascii=False).encode("utf8"))

    with open('../../data/interim/scraping/has_scraped.json','r') as json_file:
        has_scraped = json.load(json_file)
    has_scraped[link] = True
    
    with open("../../data/interim/scraping/has_scraped.json", 'wb') as f:
        f.write(json.dumps(has_scraped, indent = 4, ensure_ascii=False).encode("utf8"))
    

def get_urls(driver,link):
    driver.get(link)
    time.sleep(10)

    try:
        cookie = driver.find_element(By.ID,'onetrust-accept-btn-handler')
        cookie.click()
    except NoSuchElementException:
        pass

    while True:
        try:
            time.sleep(5)
            button = driver.find_element(By.CLASS_NAME,'show-more-button')
            button.click()
        except:
            break

    elems = driver.find_elements(By.CLASS_NAME, "feed-item")
    s = set()
    for elem in elems:
        link = elem.get_attribute('innerHTML').split('"')[1]
        s.add(link)

    with open('../../data/interim/scraping/links.txt','w') as f:
        for elem in s:
            f.write(f'tasty.co{elem}\n')

def get_tags(driver):
    li = []
    for i in driver.find_elements(By.CLASS_NAME,'breadcrumb_wrapper'):
        for j in i.find_elements(By.CLASS_NAME,'breadcrumb_item'):
            li.append(j.text)
    return li

def get_images(driver):
    try:
        os.mkdir('images')
    except FileExistsError:
        pass
    try:
        cookie = driver.find_element(By.ID,'onetrust-accept-btn-handler')
        cookie.click()

    except (NoSuchElementException, ElementNotInteractableException) :
        pass
    while True:
        try:
            time.sleep(2)
            button = driver.find_element(By.CLASS_NAME,'show-more-button')
            button.click()
            print('Pressing show more...')
        except:
            print('Reached bottom')
            break
    
    link = driver.find_elements(By.CLASS_NAME,'feed-item')
    li = {}
    for i in link:
        href = i.get_attribute('innerHTML').split('"')[1]
        s = i.find_elements(By.TAG_NAME,'img')
        for j in s:
            if 'img.buzzfeed.com' in j.get_attribute('src'):
                picture = j.get_attribute('src')
            else:
                picture = None
        li[href] = picture

    with open("../../data/interim/scraping/images_links.json", "w") as outfile: 
        json.dump(li, outfile,indent = 4,ensure_ascii=False)
    
    for i,j in li.items():
            if j != None:
                r = requests.get(j,allow_redirects=True)
                with open('../../data/images/'+str(i)[7:]+'.jpg', 'wb') as f:
                    f.write(r.content)