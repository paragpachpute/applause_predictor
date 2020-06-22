$(document).ready(function(){
	$("#reportDownload").click(function(){
		$("#downloadReportLink")[0].click();
	})

	$("#posterDownload").click(function(){
		$("#downloadPosterLink")[0].click();
	})

	$("#tryItButton").click(function(){
		console.log("hello")
		$(".tryingIt").toggleClass("active")
	})

	$("#example1").click(function(){
		$("#textbox").val("Yo Bro")
	});

	$("#example2").click(function(){
		$("#textbox").val("Yo Bro2")
	});

	$("#example3").click(function(){
		$("#textbox").val("Yo Bro3")
	});

	$("#submitButton").click(function(){
		console.log("hi")
		if ($("#textbox").val() == ""){
			alert("No text entered")
			return
		}

		$(".submitButton").addClass("loading")
		var settings = {
		  "url": "http://localhost:5000/detect",
		  "method": "POST",
		  "timeout": 0,
		  "headers": {
		    "Content-Type": "application/json"
		  },
		  "data": JSON.stringify({"sentences":[$("#textbox").val()]}),
		};

		$.ajax(settings).done(function (response) {
		  if (response.pred == "Negative"){
		  	$("#pred").html("No Applause!")
		  	$("#conf").html(Math.round(response.confidence[0][1] * 100) / 100)
		  } else {
			$("#pred").html("Applause!")
			$("#conf").html(Math.round(response.confidence[0][0] * 100) / 100)
		  }
		  $(".results").removeClass("hidden")
		  $(".submitButton").removeClass("loading")
		});	

	})

})