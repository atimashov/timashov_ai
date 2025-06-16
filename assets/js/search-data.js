// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-repositories",
          title: "Repositories",
          description: "My opensource playground",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-speaking",
          title: "Speaking",
          description: "I enjoy making complex AI topics accessible — from Semi-Supervised Learning to Computer Vision — through talks at global conferences and developer communities.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/speaking/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "My CV reflects 10+ years of AI impact across the UK and Southeast Asia, and a current focus on building research-grounded systems with real-world reach.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-happy-to-share-that-i-am-invited-to-facilitate-stanford-s-machine-learning-with-graphs-course-xcs224w-as-part-of-the-ai-professional-program-the-new-cohort-begins-october-7th-2024",
          title: 'Happy to share that I am invited to facilitate Stanford’s “Machine Learning with...',
          description: "",
          section: "News",},{id: "news-i-am-invited-to-facilitate-stanford-s-xcs224w-machine-learning-with-graphs-course-again-happy-to-support-professionals-globally",
          title: 'I am invited to facilitate Stanford’s XCS224W “Machine Learning with Graphs” course again....',
          description: "",
          section: "News",},{id: "speaking-big-data-ldn",
          title: 'Big Data LDN',
          description: "Talk on Semi-Supervised Learning for Object Detection in London (UK).",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/big_data_ldn/";
            },},{id: "speaking-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/1_project/";
            },},{id: "speaking-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/2_project/";
            },},{id: "speaking-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/3_project/";
            },},{id: "speaking-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/4_project/";
            },},{id: "speaking-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/5_project/";
            },},{id: "speaking-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/6_project/";
            },},{id: "speaking-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/7_project/";
            },},{id: "speaking-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/examples/8_project/";
            },},{id: "speaking-beyond-the-code",
          title: 'Beyond the code',
          description: "Master class in Semi-Supervised Learning for &quot;Microsoft Learn Student Ambassadors Nigeria&quot;.",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/ssl_nigeria/";
            },},{id: "speaking-apply-by-tecton",
          title: 'apply() by Tecton',
          description: "Talk of Semi-Supervised Learning online.",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/tecton/";
            },},{id: "speaking-wad-world-congress",
          title: 'WAD World Congress',
          description: "Talk on Semi-Supervised Learning at &quot;WeAreDevelopers World Congress&quot; in Berlin (Germany).",
          section: "Speaking",handler: () => {
              window.location.href = "/speaking/world_congress/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
