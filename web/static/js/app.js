$(function () {
  var basePath = '/results';
  var resultsUrl = '/results';
  var groupUrl = '/results/:groupid';
  var expUrl = '/results/:groupid/:name';
  var generatePlotUrl = '/generate_plot';

  var currMetrics = null;
  var currExpUrl = null;

  var $pathBreadcrumbs = $('#path_container ol');
  var $pathText = $('#path_container h3');
  var $contentOutterContainer = $('#content_outter_container');
  var $contentContainer = $('#content_container');
  var $contentContainerTitle = $('#content_container_title');
  var $entriesList = $('#path_selection_container #content_entries');
  var $resultViewer = $('#result_viewer');
  var $plotPlaceholder = $('#plot_placeholder');
  var $plotCreateButton = $('#content_outter_container form button');
  var $resetButton = $('#reset_button');
  var $metricsCombo = $('#metrics_combo');
  var $plotType = $('#plot_type');
  var $resultsLink = $('#results_link');
  var $plotModal = $('#plot_modal');

  var loadGroups = function () {
    $contentOutterContainer.hide();

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

        $pathBreadcrumbs.find('li.removable').remove();
      });
    });
  };

  loadGroups();

  var loadExp = function (groupid, expName) {
    var exactExpUrl = expUrl.replace(':groupid', groupid)
                            .replace(':name', expName);

    $.get(exactExpUrl, function (metrics) {
      currMetrics = metrics;
      currExpUrl = exactExpUrl;
      setBreadcrumbs(groupid, expName);
      $resultViewer.jsonViewer(metrics, {collapsed: true});
      $contentOutterContainer.find('h5').text(groupid + '/' + expName);
      $contentOutterContainer.show();
    });
  };

  var loadGroup = function (groupid) {
    $contentOutterContainer.hide();
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

      setBreadcrumbs(groupid);
    });
  };

  var $breadcrumbLinkTmpl = $('<a />').attr('href', '#');
  var $breadcrumbTmpl = $('<li />').addClass('breadcrumb-item')
                                   .addClass('removable');

  var setBreadcrumbs = function (groupid, expName) {
    $pathBreadcrumbs.find('li.removable').remove()

    var $newBreadcrumbGroup = $breadcrumbTmpl.clone();
    var $newBreadcrumbGroupLink = $breadcrumbLinkTmpl.clone();

    $newBreadcrumbGroupLink.text(groupid);
    $newBreadcrumbGroupLink.click(function (e) {
      e.preventDefault();
      loadGroup(groupid);
    });

    $newBreadcrumbGroupLink.appendTo($newBreadcrumbGroup);
    $newBreadcrumbGroup.appendTo($pathBreadcrumbs);

    var $newBreadcrumbExp = $breadcrumbTmpl.clone();
    var $newBreadcrumbExpLink = $breadcrumbLinkTmpl.clone();

    $newBreadcrumbExpLink.text(expName);
    $newBreadcrumbExpLink.click(function (e) {
      e.preventDefault();
      loadExp(groupid, expName);
    });

    $newBreadcrumbExpLink.appendTo($newBreadcrumbExp);
    $newBreadcrumbExp.appendTo($pathBreadcrumbs);
  };

  $resultsLink.click(function () {
    loadGroups();
  });

  $plotCreateButton.click(function (e) {
    e.preventDefault();

    var metrics = $metricsCombo.val();

    if (!currExpUrl) {
      alert('No experiment selected!');
      return;
    }

    if (metrics.length == 0) {
      alert('No metrics selected!');
      return;
    }

    var fullName = currExpUrl.replace('/results/', '');
    var params = {
      'metrics': $metricsCombo.val().join(','),
      'plot_type': $plotType.val(),
      'full_name': fullName
    };

    var plotUrl = generatePlotUrl + '?' + $.param(params);
    $plotPlaceholder.attr('src', plotUrl);
    $plotModal.modal('show');
  });
});