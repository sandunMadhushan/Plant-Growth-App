module.exports = {
    content: [".//*.html"],
    theme: {
      extend: {},
    },
    plugins: [],
  }


module.exports = {
    content: ['./templates/**/*.{html,js}', './static/**/*.{js,css}'], 
    theme: {
      extend: {
        colors: {
          ecoGreen: '#006633', 
        },
      },
    },
    plugins: [],
  }

  module.exports = {
    content: ['./templates/**/*.{html,js}', './static/**/*.{js,css}'], 
    theme: {
      extend: {
        colors: {
          bgreen: '#00351B',
        },
      },
    },
    plugins: [],
  }

  // tailwind.config.js
module.exports = {
    theme: {
      extend: {
        fontFamily: {
          inter: ['Inter', 'sans-serif'],
        },
      },
    },
    // Don't forget your content paths
    content: ['./templates/**/*.{html,js}', './static/**/*.{js,css}'],
  }
  
  