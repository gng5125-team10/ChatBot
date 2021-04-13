from bs4 import BeautifulSoup
import csv
import requests
import pandas as pd
import json
import re

def TextAfterTag(text, tag):
    pos = text.find(tag)
    if pos != -1:
        return text[pos+len(tag):]
    return text

def removeDuplicates(mylist):
    return list( dict.fromkeys(mylist) )

def getPageThreadsIds(url):
    page_ids = []
    page = requests.get(url)
    html = page.text
    soup = BeautifulSoup(html, "html.parser")
    for row in soup.find_all('tr', attrs={'class': 'inline_row'}):
        res = re.search("tid=([0-9]+)", str(row))
        if res:
            id = res.group(0)
            page_ids.append(id)
    return page_ids


def getPagesInAThread(url):
    pages = [url]
    page = requests.get(url)
    html = page.text
    #print(html)
    url_including_page = url + "&page="
    soup = BeautifulSoup(html, "html.parser")
    max_pages =  soup.find('span', attrs={'class': 'pages'})
    if max_pages:
        #print(max_pages.text)
        m = max_pages.text.replace("Pages (", "").replace("):","")
        for page_num in range(2, int(m)):
          pages.append(url_including_page+str(page_num))
    return pages

#scrap data from forum data page
def scrapPageData(url):

    #print("scrapping page:"+url)
    # Connect to the URL and download document
    page = requests.get(url)
    html = page.text
    soup = BeautifulSoup(html, "html.parser")
    
    posts = []
    responses  = []

    first_post = True
    for fond_post in soup.find_all('div', attrs={'class': 'post_body scaleimages'}):
        post = fond_post.text.strip()
        post = post.replace("ADVERTISEMENTS", "")
        #print(post)
        if post == "":
            continue

        if first_post:
            original_question = post  # this is the first post in a given thread
            first_post = False
        else:
        
            if "Wrote:" in post: #text contain quotes to previous text
                for quote in fond_post.find_all('blockquote', attrs={'class': 'mycode_quote'}):
                    Q = quote.text
                    #find text after quote
                    pos = post.find(Q)
                    if pos != -1:
                        response_pos = pos + len(Q)
                        A = post[response_pos:].strip()                  
                        #Clear "Wrote:"
                        Q = TextAfterTag(Q,"Wrote:").strip()
                        #output QA    
                        if len(Q)> 0 and len(A)>0:
                            posts.append(Q)
                            responses.append(A)
            else:
                #the post is response to the first post
                q = original_question
                response = post
                posts.append(q)
                responses.append(response)
        
    # Create an empty data frame
    df = pd.DataFrame()
    
    # Add a posts to df
    df['Posts'] = posts
    df['Responses'] = responses

    print("url:" +url +" posts: "+str(len(df)))

    return df

# get page source and create a BeautifulSoup object based on it
#url=("https://www.alonelylife.com/showthread.php?tid=40300")
#url='https://www.alonelylife.com/showthread.php?tid=41135'

forum_url = 'https://www.alonelylife.com/forumdisplay.php?fid=4'
base_thread_url= 'https://www.alonelylife.com/showthread.php?'
page_range = range(200,230)
df = pd.DataFrame()
threads_ids = []
for page_num in page_range:
    #print("Page:"+str(page_num))
    url = forum_url+"&page="+str(page_num)
    ids = getPageThreadsIds(url)
    threads_ids.extend(ids)

removeDuplicates(threads_ids)
#print(threads_ids)
thread_pages_urls=[]
for id in threads_ids:
    urls = getPagesInAThread(base_thread_url+str(id))
    thread_pages_urls.extend(urls)

#print (pages)
df = pd.DataFrame()
for url in thread_pages_urls:
    page_df = scrapPageData(url)
    df = df.append(page_df, ignore_index=True)

print(df)
print(len(df))

## 3. Save data
df.to_pickle("./loneliness_forum_data_200_230.pkl")