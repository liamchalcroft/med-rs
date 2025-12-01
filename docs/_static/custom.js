// Custom JavaScript for medrs documentation

// =============================================
// Performance Metrics Interactive Elements
// =============================================

document.addEventListener('DOMContentLoaded', function() {

  // Animate performance metrics on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
      }
    });
  }, observerOptions);

  // Observe all cards and tables
  document.querySelectorAll('.feature-card, .quickstart-card, .performance-table').forEach(el => {
    observer.observe(el);
  });

  // =============================================
  // Performance Comparison Interactive
  // =============================================

  const performanceComparison = document.querySelector('.performance-comparison');
  if (performanceComparison) {
    // Add hover effects to table rows
    const rows = performanceComparison.querySelectorAll('tbody tr');
    rows.forEach(row => {
      row.addEventListener('mouseenter', function() {
        this.style.backgroundColor = '#f0f9ff';
        this.style.transform = 'scale(1.02)';
        this.style.transition = 'all 0.2s ease';
      });

      row.addEventListener('mouseleave', function() {
        this.style.backgroundColor = '';
        this.style.transform = '';
      });
    });
  }

  // =============================================
  // Code Copy Button Enhancement
  // =============================================

  function enhanceCopyButtons() {
    // Style copy buttons
    const style = document.createElement('style');
    style.textContent = `
      .copy-button {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        background: var(--medrs-primary);
        color: white;
        border: none;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s ease;
      }

      .highlight:hover .copy-button {
        opacity: 1;
      }

      .copy-button:hover {
        background: var(--medrs-primary-dark);
      }

      .copy-button.success {
        background: var(--medrs-success);
      }
    `;
    document.head.appendChild(style);

    // Add copy buttons to code blocks
    document.querySelectorAll('.highlight').forEach(block => {
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy code block');

      block.style.position = 'relative';
      block.appendChild(button);

      button.addEventListener('click', async () => {
        const code = block.querySelector('code') || block;
        const text = code.textContent;

        try {
          await navigator.clipboard.writeText(text);
          button.textContent = 'Copied!';
          button.classList.add('success');

          setTimeout(() => {
            button.textContent = 'Copy';
            button.classList.remove('success');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy text: ', err);
          button.textContent = 'Failed';
          setTimeout(() => {
            button.textContent = 'Copy';
          }, 2000);
        }
      });
    });
  }

  // =============================================
  // Search Enhancement
  =============================================

  function enhanceSearch() {
    const searchInput = document.querySelector('#search-input');
    if (searchInput) {
      searchInput.setAttribute('placeholder', 'Search medrs documentation...');

      // Add search suggestions
      searchInput.addEventListener('focus', () => {
        showSearchSuggestions();
      });

      searchInput.addEventListener('blur', () => {
        hideSearchSuggestions();
      });
    }
  }

  function showSearchSuggestions() {
    const suggestions = [
      'crop-first loading',
      'performance optimization',
      'MONAI integration',
      'PyTorch tensors',
      'memory efficiency',
      'async training'
    ];

    const suggestionBox = document.createElement('div');
    suggestionBox.className = 'search-suggestions';
    suggestionBox.style.cssText = `
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: white;
      border: 1px solid #e2e8f0;
      border-top: none;
      border-radius: 0 0 0.25rem 0.25rem;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 1000;
      max-height: 200px;
      overflow-y: auto;
    `;

    suggestions.forEach(suggestion => {
      const item = document.createElement('div');
      item.textContent = suggestion;
      item.style.cssText = `
        padding: 0.5rem 1rem;
        cursor: pointer;
        border-bottom: 1px solid #f1f5f9;
        transition: background-color 0.2s ease;
      `;

      item.addEventListener('mouseenter', () => {
        item.style.backgroundColor = '#f8fafc';
      });

      item.addEventListener('mouseleave', () => {
        item.style.backgroundColor = '';
      });

      item.addEventListener('click', () => {
        const searchInput = document.querySelector('#search-input');
        searchInput.value = suggestion;
        searchInput.dispatchEvent(new Event('input'));
        hideSearchSuggestions();
      });

      suggestionBox.appendChild(item);
    });

    const searchContainer = document.querySelector('.bd-search-container');
    if (searchContainer) {
      searchContainer.style.position = 'relative';
      searchContainer.appendChild(suggestionBox);
    }
  }

  function hideSearchSuggestions() {
    const suggestions = document.querySelector('.search-suggestions');
    if (suggestions) {
      suggestions.remove();
    }
  }

  // =============================================
  // TOC Auto-Generation for Long Pages
  =============================================

  function generateTOC() {
    const content = document.querySelector('.bd-content');
    if (!content) return;

    const headings = content.querySelectorAll('h2, h3, h4, h5, h6');
    if (headings.length < 5) return; // Only generate for long pages

    const tocContainer = document.createElement('div');
    tocContainer.className = 'page-toc';
    tocContainer.innerHTML = `
      <h4>Page Contents</h4>
      <nav class="page-toc-nav"></nav>
    `;

    const tocNav = tocContainer.querySelector('.page-toc-nav');

    let currentLevel = 2;
    let currentList = null;

    headings.forEach((heading, index) => {
      const level = parseInt(heading.tagName.charAt(1));
      const text = heading.textContent;
      const id = heading.id || `heading-${index}`;

      if (!heading.id) {
        heading.id = id;
      }

      // Adjust list nesting
      if (level > currentLevel) {
        while (level > currentLevel) {
          const nestedList = document.createElement('ul');
          if (currentList && currentList.lastElementChild) {
            currentList.lastElementChild.appendChild(nestedList);
            currentList = nestedList;
          } else {
            if (!currentList) {
              currentList = document.createElement('ul');
              tocNav.appendChild(currentList);
            }
            currentList.appendChild(nestedList);
            currentList = nestedList;
          }
          currentLevel++;
        }
      } else if (level < currentLevel) {
        while (level < currentLevel && currentList.parentElement) {
          currentList = currentList.parentElement;
          currentLevel--;
        }
      }

      // Create list if needed
      if (!currentList) {
        currentList = document.createElement('ul');
        tocNav.appendChild(currentList);
        currentLevel = level;
      }

      // Create list item
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = `#${id}`;
      link.textContent = text;
      link.className = `toc-level-${level}`;

      item.appendChild(link);
      currentList.appendChild(item);
    });

    // Insert TOC after first paragraph
    const firstParagraph = content.querySelector('p');
    if (firstParagraph) {
      firstParagraph.parentNode.insertBefore(tocContainer, firstParagraph.nextSibling);
    } else {
      content.insertBefore(tocContainer, content.firstChild);
    }
  }

  // =============================================
  // Implementation Status Indicators
  // =============================================

  function addImplementationStatus() {
    const statusIndicators = {
      '': 'completed',
      '': 'in-progress',
      '': 'planned',
      '': 'reviewed'
    };

    document.querySelectorAll('p, li, td').forEach(element => {
      const text = element.textContent;
      let modified = false;

      Object.entries(statusIndicators).forEach(([emoji, status]) => {
        if (text.includes(emoji)) {
          const span = document.createElement('span');
          span.className = `implementation-status ${status}`;
          span.textContent = emoji;
          span.setAttribute('title', `Status: ${status.replace('-', ' ')}`);

          element.textContent = text.replace(emoji, '');
          element.insertBefore(span, element.firstChild);
          modified = true;
        }
      });
    });
  }

  // =============================================
  // Performance Metrics Animation
  // =============================================

  function animateMetrics() {
    const metrics = document.querySelectorAll('.performance-metric');

    metrics.forEach((metric, index) => {
      metric.style.opacity = '0';
      metric.style.transform = 'translateY(20px)';

      setTimeout(() => {
        metric.style.transition = 'all 0.5s ease';
        metric.style.opacity = '1';
        metric.style.transform = 'translateY(0)';
      }, index * 100);
    });
  }

  // =============================================
  // Keyboard Shortcuts
  // =============================================

  function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + K for search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('#search-input');
        if (searchInput) {
          searchInput.focus();
        }
      }

      // ESC to clear search
      if (e.key === 'Escape') {
        const searchInput = document.querySelector('#search-input');
        if (searchInput) {
          searchInput.value = '';
          searchInput.blur();
          hideSearchSuggestions();
        }
      }

      // Ctrl/Cmd + / for search
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        const searchInput = document.querySelector('#search-input');
        if (searchInput) {
          searchInput.focus();
        }
      }
    });
  }

  // =============================================
  // Initialize all enhancements
  // =============================================

  // Initialize enhancements
  enhanceCopyButtons();
  enhanceSearch();
  setupKeyboardShortcuts();

  // Generate TOC for long pages after content loads
  if (document.readyState === 'complete') {
    generateTOC();
    addImplementationStatus();
    animateMetrics();
  } else {
    window.addEventListener('load', () => {
      generateTOC();
      addImplementationStatus();
      animateMetrics();
    });
  }

  // =============================================
  // Print-Friendly Enhancements
  =============================================

  window.addEventListener('beforeprint', () => {
    // Hide interactive elements for printing
    const elementsToHide = document.querySelectorAll(
      '.copy-button, .search-suggestions, .bd-sidebar-secondary, .bd-footer'
    );
    elementsToHide.forEach(el => {
      el.style.display = 'none';
    });

    // Show all hidden content
    const elementsToShow = document.querySelectorAll(
      '[data-print-hidden="false"]'
    );
    elementsToShow.forEach(el => {
      el.style.display = '';
    });
  });

  window.addEventListener('afterprint', () => {
    // Restore hidden elements
    const elementsToRestore = document.querySelectorAll(
      '.copy-button, .search-suggestions, .bd-sidebar-secondary, .bd-footer'
    );
    elementsToRestore.forEach(el => {
      el.style.display = '';
    });
  });

  // =============================================
  // Dark Mode Toggle
  // =============================================

  function addDarkModeToggle() {
    const darkModeToggle = document.createElement('button');
    darkModeToggle.innerHTML = '';
    darkModeToggle.className = 'dark-mode-toggle';
    darkModeToggle.setAttribute('aria-label', 'Toggle dark mode');
    darkModeToggle.style.cssText = `
      position: fixed;
      top: 1rem;
      right: 1rem;
      background: var(--medrs-primary);
      color: white;
      border: none;
      padding: 0.5rem;
      border-radius: 50%;
      cursor: pointer;
      z-index: 1000;
      font-size: 1rem;
      transition: all 0.2s ease;
    `;

    const setToggleLabel = (isDark) => {
      darkModeToggle.innerHTML = isDark ? 'Light' : 'Dark';
    };

    darkModeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      const isDark = document.body.classList.contains('dark-mode');
      setToggleLabel(isDark);

      // Save preference
      localStorage.setItem('dark-mode', isDark ? 'true' : 'false');
    });

    // Load saved preference
    const savedMode = localStorage.getItem('dark-mode');
    if (savedMode === 'true') {
      document.body.classList.add('dark-mode');
      setToggleLabel(true);
    } else {
      setToggleLabel(false);
    }

    document.body.appendChild(darkModeToggle);
  }

  addDarkModeToggle();
});
