---
layout: page
title: Speaking
permalink: /speaking/
description: I enjoy making complex AI topics accessible — from Semi-Supervised Learning to Computer Vision — through talks at global conferences and developer communities.
nav: true
nav_order: 4
display_categories: [Work] # , Fun]
horizontal: false # true
---

<!-- pages/speaking.md -->
<div class="speaking">
{% if site.enable_speaking_categories and page.display_categories %}
  <!-- Display categorized speaking -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_speaking = site.speaking | where: "category", category %}
  {% assign sorted_speaking = categorized_speaking | sort: "importance" %}
  <!-- Generate cards for each speaking -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for speaking in sorted_speaking %}
      {% include speaking_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for speaking in sorted_speaking %}
      {% include speaking.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display speaking without categories -->

{% assign sorted_speaking = site.speaking | sort: "importance" %}

  <!-- Generate cards for each speaking -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for speaking in sorted_speaking %}
      {% include speaking_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for speaking in sorted_speaking %}
      {% include speaking.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
