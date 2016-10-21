$(function () {
  var basePath = '/results';
  var resultsUrl = '/results';
  var groupUrl = '/results/:groupid';
  var expUrl = '/results/:groupid/:name';
  var generatePlotUrl = '/generate_plot';

  var currMetrics = null;
  var currExpUrl = null;

  var $pathText = $('#path_container h3');
  var $entriesList = $('#content_container ul');
  var $loadingText = $('#content_container h3');
  var $resultViewer = $('#result_viewer');
  var $plotPlaceholder = $('#plot_placeholder');
  var $plotCreateButton = $('#result_plots button');
  var $resetButton = $('#reset_button');
  var $metricsCombo = $('#metrics_combo');
  var $plotType = $('#plot_type');

  var loadExp = function (groupid, expName) {
    var exactExpUrl = expUrl.replace(':groupid', groupid)
                             .replace(':name', expName);

    $.get(exactExpUrl, function (metrics) {
      currMetrics = metrics;
      currExpUrl = exactExpUrl;

      $resultViewer.jsonViewer(metrics, {collapsed: true});
    });
  }

  var loadGroup = function (groupid) {
    var currGroupUrl = groupUrl.replace(':groupid', groupid);

    $.get(currGroupUrl, function (exps) {
      $pathText.text(basePath + '/' + groupid);

      $entriesList.find('li').remove();

      $.each(exps, function () {
        var $newEntry = $('<li />');
        var $newEntryLink = $('<a />');
        var expName = this;

        $newEntryLink.text(expName);
        $newEntryLink.click(function () {
          loadExp(groupid, expName);
        });

        $newEntryLink.appendTo($newEntry);
        $newEntry.appendTo($entriesList);
      });
    });
  };

  $.get(resultsUrl, function (groups) {
    $entriesList.find('li').remove();

    $.each(groups, function () {
      var $newEntry = $('<li />');
      var $newEntryLink = $('<a />');
      var groupid = this;

      $newEntryLink.text(groupid);
      $newEntryLink.click(function () {
        loadGroup(groupid);
      });

      $newEntryLink.appendTo($newEntry);
      $newEntry.appendTo($entriesList);
    });
  });

  $plotCreateButton.click(function () {
    if (!currExpUrl) {
      alert('No experiment selected!');
    }

    var fullName = currExpUrl.replace('/results/', '');
    var params = {
      'metrics': $metricsCombo.val().join(','),
      'plot_type': $plotType.val(),
      'full_name': fullName
    };

    var plotUrl = generatePlotUrl + '?' + $.param(params);
    $plotPlaceholder.attr('src', plotUrl);
  });

  $resetButton.click(function () {
    window.location.reload();
  });
});