#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bs4 import BeautifulSoup
import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')

svlist = [
    "http://video.fc2.com/content/",
    "http://www.miomio.tv",
    "http://www.anitube.se/video/",
    "http://video.9tsu",
    "http://www.dailymotion.com/video",
    "http://channel.pandora.tv",
    "https://www.youtube.com/"
    "http://say-move.org/comeplay",
    "http://up.b9dm.com/",
    "streamin",
    # "abc","abcnews","abcotvs","academicearth","acast","addanime","adobepass","adobetv","adultswim","aenetworks","afreecatv","aftonbladet","airmozilla","aljazeera","allocine","alphaporno","amcnetworks","amp","animeondemand","anitube","anvato","anysex","aol","aparat","appleconnect","appletrailers","archiveorg","ard","arkena","arte","atresplayer","atttechchannel","audimedia","audioboom","audiomack","awaan","azubu","baidu","bambuser","bandcamp","bbc","beatportpro","beeg","behindkink","bellmedia","bet","bigflix","bild","bilibili","biobiochiletv","biqle","bleacherreport","blinkx","bloomberg","bokecc","bpb","br","bravotv","breakcom","brightcove","buzzfeed","byutv","c56","camdemy","camwithher","canalc2","canalplus","canvas","carambatv","cartoonnetwork","cbc","cbs","cbsinteractive","cbslocal","cbsnews","cbssports","ccc","cctv","cda","ceskatelevize","channel9","charlierose","chaturbate","chilloutzone","chirbit","cinchcast","clipfish","cliphunter","cliprs","clipsyndicate","closertotruth","cloudy","clubic","clyp","cmt","cnbc","cnn","collegerama","comcarcoff","comedycentral","common","commonmistakes","commonprotocols","condenast","coub","cracked","crackle","criterion","crooksandliars","crunchyroll","cspan","ctsnews","ctvnews","cultureunplugged","curiositystream","cwtv","dailymail","dailymotion","daum","dbtv","dctp","deezer","defense","democracynow","dfb","dhm","digiteka","discovery","discoverygo","dispeak","dotsub","douyutv","dplay","dramafever","drbonanza","dreisat","dropbox","drtuber","drtv","dumpert","dvtv","dw","eagleplatform","ebaumsworld","echomsk","ehow","eighttracks","einthusan","eitb","ellentv","elpais","embedly","engadget","eporner","eroprofile","escapist","espn","esri","europa","everyonesmixtape","expotv","extractors","extremetube","eyedotv","facebook","faz","fc2","fczenit","firstpost","firsttv","fivemin","fivetv","fktv","flickr","flipagram","folketinget","footyroom","formula1","fourtube","fox","foxgay","foxnews","foxsports","franceculture","franceinter","francetv","freesound","freespeech","freevideo","funimation","funnyordie","fusion","fxnetworks","gameinformer","gameone","gamersyde","gamespot","gamestar","gazeta","gdcvault","generic","gfycat","giantbomb","giga","glide","globo","go","godtube","godtv","golem","googledrive","googleplus","googlesearch","goshgay","gputechconf","groupon","hark","hbo","hearthisat","heise","hellporno","helsinki","hentaistigma","hgtv","historicfilms","hitbox","hornbunny","hotnewhiphop","hotstar","howcast","howstuffworks","hrti","huffpost","hypem","iconosquare","ign","imdb","imgur","ina","indavideo","infoq","instagram","internetvideoarchive","iprima","iqiyi","ir90tv","ivi","ivideon","iwara","izlesene","jeuxvideo","jove","jpopsukitv","jwplatform","kaltura","kamcord","kanalplay","kankan","karaoketv","karrierevideos","keek","keezmovies","ketnet","khanacademy","kickstarter","konserthusetplay","kontrtube","krasview","ku6","kusi","kuwo","la7","laola1tv","lci","lcp","learnr","lecture2go","leeco","lemonde","libraryofcongress","libsyn","lifenews","limelight","litv","liveleak","livestream","lnkgo","localnews8","lovehomeporn","lrt","lynda","m6","macgamestore","mailru","makerschannel","makertv","mangomolo","matchtv","mdr","meta","metacafe","metacritic","mgoon","mgtv","miaopai","microsoftvirtualacademy","minhateca","ministrygrid","minoto","miomio","mit","mitele","mixcloud","mlb","mnet","moevideo","mofosex","mojvideo","moniker","morningstar","motherless","motorsport","movieclips","moviezine","movingimage","mpora","msn","mtv","muenchentv","musicplayon","mwave","myspace","myspass","myvi","myvideo","myvidster","nationalgeographic","naver","nba","nbc","ndr","ndtv","nerdcubed","neteasemusic","netzkino","newgrounds","newstube","nextmedia","nfb","nfl","nhk","nhl","nick","niconico","ninecninemedia","ninegag","ninenow","nintendo","noco","normalboots","nosvideo","nova","novamov","nowness","nowtv","noz","npo","npr","nrk","ntvde","ntvru","nuevo","nuvid","nytimes","odatv","odnoklassniki","oktoberfesttv","once","onet","onionstudios","ooyala","openload","ora","orf","pandoratv","parliamentliveuk","patreon","pbs","people","periscope","philharmoniedeparis","phoenix","photobucket","pinkbike","pladform","playfm","plays","playtvak","playvid","playwire","pluralsight","podomatic","pokemon","polskieradio","porn91","porncom","pornhd","pornhub","pornotube","pornovoisines","pornoxo","presstv","primesharetv","promptfile","prosiebensat1","puls4","pyvideo","qqmusic","r7","radiobremen","radiocanada","radiode","radiofrance","radiojavan","rai","rbmaradio","rds","redtube","regiotv","restudy","reuters","reverbnation","revision3","rice","ringtv","rmcdecouverte","ro220","rockstargames","roosterteeth","rottentomatoes","roxwel","rozhlas","rtbf","rte","rtl2","rtlnl","rtp","rts","rtve","rtvnh","rudo","ruhd","ruleporn","rutube","rutv","ruutu","safari","sandia","sapo","savefrom","sbs","scivee","screencast","screencastomatic","screenjunkies","screenwavemedia","seeker","senateisvp","sendtonews","servingsys","sexu","shahid","shared","sharesix","sina","sixplay","skynewsarabia","skysports","slideshare","slutload","smotri","snotr","sohu","sonyliv","soundcloud","soundgasm","southpark","spankbang","spankwire","spiegel","spiegeltv","spike","sport5","sportbox","sportdeutschland","sportschau","srgssr","srmediathek","stanfordoc","steam","stitcher","streamable","streamcloud","streamcz","streetvoice","sunporno","svt","swrmediathek","syfy","sztvhu","tagesschau","tass","tbs","tdslifeway","teachertube","teachingchannel","teamcoco","techtalks","ted","tele13","telebruxelles","telecinco","telegraaf","telemb","telequebec","teletask","telewebion","testurl","tf1","tfo","theintercept","theplatform","thescene","thesixtyone","thestar","thisamericanlife","thisav","threeqsdn","tinypic","tlc","tmz","tnaflix","toggle","toutv","toypics","traileraddict","trilulilu","trutv","tube8","tubitv","tudou","tumblr","tunein","turbo","turner","tutv","tv2","tv3","tv4","tvc","tvigle","tvland","tvnoe","tvp","tvplay","tweakers","twentyfourvideo","twentymin","twentytwotracks","twitch","twitter","udemy","udn","unistra","uol","uplynk","urort","urplay","usanetwork","usatoday","ustream","ustudio","varzesh3","vbox7","veehd","veoh","vessel","vesti","vevo","vgtv","vh1","vice","viceland","vidbit","viddler","videodetective","videofyme","videomega","videomore","videopremium","videott","vidio","vidme","vidzi","vier","viewlift","viewster","viidea","viki","vimeo","vimple","vine","vk","vlive","vodlocker","vodplatform","voicerepublic","voxmedia","vporn","vrt","vube","vuclip","vyborymos","walla","washingtonpost","wat","watchindianporn","wdr","webofstories","weiqitv","wimp","wistia","worldstarhiphop","wrzuta","wsj","xbef","xboxclips","xfileshare","xhamster","xiami","xminus","xnxx","xstream","xtube","xuite","xvideos","xxxymovies","yahoo","yam","yandexmusic","yesjapan","yinyuetai","ynet","youjizz","youku","youporn","yourupload","youtube","zapiks","zdf","zingmp3"
]


def list_in_list(checklis, st):
    try:
        for check in checklis:
            if check in st:
                return st
    except:
        return None


def get_soup_uselenium(url):
    from selenium import webdriver
    # need chromedriver #https://sites.google.com/a/chromium.org/chromedriver/downloads
    # chromedriver = "./chromedriver"
    # driver = webdriver.Chrome(chromedriver)
    driver = webdriver.PhantomJS(service_log_path=os.path.devnull)
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "lxml")
    driver.quit()
    return soup


def get_downloadlink_from_page(url, servicelist):

    def arrangeurl(urllist, svlist):
        temp = []
        for tar in svlist:
            for url in urllist:
                if tar in url:
                    temp.append(url)
        return temp
    soup = get_soup_uselenium(url)

    try:
        title = soup.find("h1").text
    except:
        title = "None"

    A = []
    for link in soup.findAll("a"):
        url = link.get("href")
        url = list_in_list(servicelist, url)
        if url:
            A.append(url)  # [:50],"..."
# if "miomio" in link.get("href"): # "mp4"を含むリンクを抽出
        # print #link.get('href')
    A = arrangeurl(A, servicelist)
    return A, title


def download_video(url_):
    print ("downloading..." + url_)

    import os
    DLlist = os.path.dirname(os.path.abspath(sys.argv[0])) + "/.downloadlist"
    f = open(DLlist)

    check = True
    try:
        sys.argv[2]
        check = False
    except:
        pass

    if os.system("cat %s| grep -o %s" % (DLlist, url_)) == 0 and check:
        print ("#ERROR: This video may be downloaded already.")
        return 0
    f.close()

    lis, title = get_downloadlink_from_page(url_, svlist)
    import youtube_dl

    class MyLogger(object):

        def debug(self, msg):
            sys.stderr.write('\r\033[K' + msg)
            sys.stderr.flush()
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    ydl_opts = {
        'outtmpl': '/Users/shun-sa/Downloads/%(title)s.%(ext)s', 'logger': MyLogger()}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for url in lis:
            try:
                ydl.download([url])
                f = open(DLlist, "a+")
                f.write(url_)
                f.close()
                print ()
                print ("Download complete!")
                return True
            except:
                pass
    print ()
    print ("Valid video url was not found.")
    return False


def get_ulist_from_channnel(channelurl):
    if "com/?q=" in channelurl:
        pass
    elif "category" not in channelurl:
        raise Exception("This url is not category page. url:%s" % channelurl)
    soup = get_soup_uselenium(channelurl)
    print soup

    urls = soup.findAll("div", {"class": "mainEntryTitle"})
    B = [i.find("b").text for i in urls]
    urls = soup.findAll("div", {"class": "mainEntryMore"})
    A = [i.find("a").get("href") for i in urls]
    return A, B


def get_animeurl_list_from_channel(channelurl):
    if "http://tvanimedouga.blog93.fc2.com" not in channelurl:
        raise Exception(
            "This url is not animecategory page. url:%s" % channelurl)

    soup = get_soup_uselenium(channelurl)

    urls = soup.find("div", {"class": "mainEntrykiji"}).findAll("a")

    A = [i.get("href") for i in urls]
    B = [i.text for i in urls]

    if not all([v is None for v in [list_in_list(["http://himado.in/?keyword", "http://www.dailymotion.com/jp/relevance/search", "https://www.youtube.com/results?search_query", "http://www.anitube.se/search/?search_id"], i) for i in A]]):
        raise Exception(
            "This url is not animecategory page. url:%s" % channelurl)
    return A, B


def download_video_mp(urllist, maxprocess=4):
    if len(urllist[0]) == 1:
        download_video(urllist)
        return True
    return [download_video(i) for i in urllist]


def get_ulist_from_text(filename):
    f = open(filename)
    ulist = [i.replace("\n", "") for i in f]
    f.close()
    return ulist

try:
    url = sys.argv[1]
except:
    print ("#ERROR: specify url or .txt file.")
    exit()


if "youtubeowaraitv" in url:
    try:
        urls, titles = get_ulist_from_channnel(url)
    except:
        urls = url
elif "tvanimedouga" in url:
    try:
        urls, titles = get_animeurl_list_from_channel(url)
    except:
        urls = url
elif "http" in url:
    urls = url

else:
    urls = get_ulist_from_text(url)

if len(urls[0]) == 1:
    print ("download %d videos..." % 1)
    print urls
else:
    print ("download %d videos..." % len(urls))
    for i in urls:
        print i
download_video_mp(urls)
