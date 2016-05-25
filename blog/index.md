---
layout: page
title: Blog
excerpt: "An archive of blog posts sorted by date."
search_omit: true
---

<ul class="post-list">
{% for post in site.categories.blog %} 
  <li><table width="100%"><tr><td width="80px">
  	{% if post.icon %}<a href="{{ site.url }}{{ post.url }}"><img src="{{ site.url }}/images/{{ post.icon }}" class="bio-photo-tiny" alt="{{ post.title }}" /></a>{% endif %}
  	</td><td><article>
  		<div><a href="{{ site.url }}{{ post.url }}">{{ post.title }} <span class="entry-date"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time></span>{% if post.excerpt %} <span class="excerpt">{{ post.excerpt | remove: '\[ ... \]' | remove: '\( ... \)' | markdownify | strip_html | strip_newlines | escape_once }}</span>{% endif %}</a></div>
  	</article>
  </td></tr></table>
  </li>
{% endfor %}
</ul>
