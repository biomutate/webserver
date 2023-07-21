// <!-- typed js effect starts for Baisc-->
var typed = new Typed(".basic .typing-text", {
    strings: ["Preprocess Raw Protein/Genomic Data", "Calculate Mutational Entropy of a species", "Train ML model for Mutation Prediction"],
    loop: true,
    typeSpeed: 50,
    backSpeed: 25,
    backDelay: 500,
});
// <!-- typed js effect ends for Basic-->

// <!-- type js effect starts for FAQ -->
const toggleButtons = document.querySelectorAll(".accordion-toggle");

toggleButtons.forEach((button) => {
  button.addEventListener("click", function () {
    const content = this.parentElement.nextElementSibling;
    this.classList.toggle("open");
    content.classList.toggle("open");
  });
});
// <!-- typed js effect ends for FAQ-->

// <!-- js for the popup starts -->
function openPopup(popupType) {
    var popupTitle = document.getElementById('popup-title');
    var popupBody = document.getElementById('popup-body');
    var overlay = document.getElementById('popup-overlay');
    var container = document.getElementById('popup-container');

    if (popupType === 'Pricing') {
        // Show Pricing popup
        var priceOverlay = document.getElementById('popup-overlay-price')
        var priceContainer = document.getElementById('popup-container-price')
        priceOverlay.style.display = 'block';
        priceContainer.style.display = 'block';
    } else if (popupType === 'FAQ') {
        // Show FAQ popup
        var faqOverlay = document.getElementById('popup-overlay-faq');
        var faqContainer = document.getElementById('popup-container-faq');
        faqOverlay.style.display = 'block';
        faqContainer.style.display = 'block';
    } else if (popupType === 'Contact') {
        // Show Contact popup
        var contactOverlay = document.getElementById('popup-overlay-contact');
        var contactContainer = document.getElementById('popup-container-contact');
        contactOverlay.style.display = 'block';
        contactContainer.style.display = 'block';
    }

}

function closePopup(popupType) {

    if (popupType === 'pricing') {
        var priceOverlay = document.getElementById('popup-overlay-price');
        var priceContainer = document.getElementById('popup-container-price');
        priceOverlay.style.display = 'none';
        priceContainer.style.display = 'none';
    }

    if (popupType === 'faq') {
        var faqOverlay = document.getElementById('popup-overlay-faq');
        var faqContainer = document.getElementById('popup-container-faq');
        faqOverlay.style.display = 'none';
        faqContainer.style.display = 'none';
    }

    if (popupType === 'contact') {
        var contactOverlay = document.getElementById('popup-overlay-contact');
        var contactContainer = document.getElementById('popup-container-contact');
        contactOverlay.style.display = 'none';
        contactContainer.style.display = 'none';
    }
}
// <!-- js for the popup ends -->

// <!-- js for page views starts -->
$(document).ready(function() {
    function updatePageViews() {
        $.get('/get_page_views_count', function(data) {
            $('#pageViewsCounter').text('Total Page Views: ' + data.count);
        });
    }

    updatePageViews();
});
// <!-- js for page views ends -->

// <!-- js for carousel -->
var timer = 4000;
var i = 0;
var max = $('#c > li').length;

	$("#c > li").eq(i).addClass('active').css('left','0');
	$("#c > li").eq(i + 1).addClass('active').css('left','25%');
	$("#c > li").eq(i + 2).addClass('active').css('left','50%');
	$("#c > li").eq(i + 3).addClass('active').css('left','75%');


	setInterval(function(){

		$("#c > li").removeClass('active');

		$("#c > li").eq(i).css('transition-delay','0.25s');
		$("#c > li").eq(i + 1).css('transition-delay','0.5s');
		$("#c > li").eq(i + 2).css('transition-delay','0.75s');
		$("#c > li").eq(i + 3).css('transition-delay','1s');

		if (i < max-4) {
			i = i+4;
		}

		else {
			i = 0;
		}

		$("#c > li").eq(i).css('left','0').addClass('active').css('transition-delay','1.25s');
		$("#c > li").eq(i + 1).css('left','25%').addClass('active').css('transition-delay','1.5s');
		$("#c > li").eq(i + 2).css('left','50%').addClass('active').css('transition-delay','1.75s');
		$("#c > li").eq(i + 3).css('left','75%').addClass('active').css('transition-delay','2s');

	}, timer);