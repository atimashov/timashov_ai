---
layout: page
title: Media
permalink: /media/
description: A growing collection of your cool media.
nav: true
nav_order: 3
display_categories: [work, fun]
horizontal: false
---

<!-- pages/media.md -->
<div class="media">
{% if site.enable_media_categories and page.display_categories %}
  <!-- Display categorized media -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_media = site.media | where: "category", category %}
  {% assign sorted_media = categorized_media | sort: "importance" %}
  <!-- Generate cards for each media -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for media in sorted_media %}
      {% include media_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for media in sorted_media %}
      {% include media.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display media without categories -->

{% assign sorted_media = site.media | sort: "importance" %}

  <!-- Generate cards for each media -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for media in sorted_media %}
      {% include media_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for media in sorted_media %}
      {% include media.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
