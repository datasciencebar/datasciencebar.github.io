---
layout: page
title: Your bartenders
excerpt: "Authors of the blog."
search_omit: true
---


{% for author in site.data.authors %}
<h1 id="{{ author[0] }}">{{ author[1].name }}</h1>

<footer class="entry-meta" style="width:150px">
    {% if author[1].avatar contains 'http' %}
    	<img src="{{ author[1].avatar }}" class="bio-photo" alt="{{ author.name }} bio photo" width="150px"></a>
    {% elsif author[1].avatar %}
        <img src="{{ site.url }}/images/{{ author[1].avatar }}" class="bio-photo" alt="{{ author.name }} bio photo" width="150px" ></a>
    {% endif %}
    <span class="author vcard"><span class="fn">
	{% if author[1].web %}<a href="{{ author[1].web }}" class="author-social" target="_blank"><i class="fa fa-fw fa-internet-explorer"></i> Web</a><br />{% endif %}	
	{% if author[1].email %}<a href="mailto:{{ author[1].email }}" class="author-social" target="_blank"><i class="fa fa-fw fa-envelope-square"></i> Email</a>{% endif %}
	{% if author[1].twitter %}<br><a href="http://twitter.com/{{ author[1].twitter }}" class="author-social" target="_blank"><i class="fa fa-fw fa-twitter-square"></i> Twitter</a>{% endif %}
    {% if author[1].facebook %}<br><a href="http://facebook.com/{{ author[1].facebook }}" class="author-social" target="_blank"><i class="fa fa-fw fa-facebook-square"></i> Facebook</a>{% endif %}
    {% if author[1].google.plus %}<br><a href="http://plus.google.com/+{{ author[1].google.plus }}" class="author-social" target="_blank"><i class="fa fa-fw fa-google-plus-square"></i> Google+</a>{% endif %}
	{% if author[1].google.scholar %}<br><a href="https://scholar.google.com/citations?user={{ author[1].google.scholar }}" class="author-social" target="_blank"><i class="ai ai-fw ai-google-scholar-square"></i> G. Scholar</a>{% endif %}
	{% if author[1].linkedin %}<br><a href="http://linkedin.com/in/{{ author[1].linkedin }}" class="author-social" target="_blank"><i class="fa fa-fw fa-linkedin-square"></i> LinkedIn</a>{% endif %}
    {% if author[1].instagram %}<br><a href="http://instagram.com/{{ author[1].instagram }}" class="author-social" target="_blank"><i class="fa fa-fw fa-instagram"></i> Instagram</a>{% endif %}
    {% if author[1].tumblr %}<br><a href="http://{{ author[1].tumblr }}.tumblr.com" class="author-social" target="_blank"><i class="fa fa-fw fa-tumblr-square"></i> Tumblr</a>{% endif %}
    {% if author[1].github %}<br><a href="http://github.com/{{ author[1].github }}" class="author-social" target="_blank"><i class="fa fa-fw fa-github"></i> Github</a>{% endif %}
    {% if author[1].stackoverflow %}<br><a href="http://stackoverflow.com/users/{{ author[1].stackoverflow }}" class="author-social" target="_blank"><i class="fa fa-fw fa-stack-overflow"></i> Stackoverflow</a>{% endif %}
    {% if author[1].pinterest %}<br><a href="http://www.pinterest.com/{{ author[1].pinterest }}" class="author-social" target="_blank"><i class="fa fa-fw fa-pinterest"></i> Pinterest</a>{% endif %}
    {% if author[1].foursquare %}<br><a href="http://foursquare.com/{{ author[1].foursquare }}" class="author-social" target="_blank"><i class="fa fa-fw fa-foursquare"></i> Foursquare</a>{% endif %}
    {% if author[1].youtube %}<br><a href="https://youtube.com/user/{{ author[1].youtube }}" class="author-social" target="_blank"><i class="fa fa-fw fa-youtube-square"></i> Youtube</a>{% endif %}
    {% if author[1].weibo %}<br><a href="http://www.weibo.com/{{ author[1].weibo }}" class="author-social" target="_blank"><i class="fa fa-fw fa-weibo"></i> Weibo</a>{% endif %}
    {% if author[1].flickr %}<br><a href="http://www.flickr.com/{{ author[1].flickr }}" class="author-social" target="_blank"><i class="fa fa-fw fa-flickr"></i> Flickr</a>{% endif %}
    {% if author[1].codepen %}<br><a href="http://codepen.io/{{ author[1].codepen }}" class="author-social" target="_blank"><i class="fa fa-fw fa-codepen"></i> CodePen</a>{% endif %}
    </span></span>
</footer>
{{ author[1].longbio }}
{% endfor %}

