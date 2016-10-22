$(function () {
  var basePath = '/results';
  var resultsUrl = '/results';
  var groupUrl = '/results/:groupid';
  var expUrl = '/results/:groupid/:name';
  var generatePlotUrl = '/generate_plot';

  var currMetrics = null;
  var currExpUrl = null;
  var currEvaluatorMappings = {
    'm_train': null,
    'm_all': null,
    'm_val': null
  };

  var $pathBreadcrumbs = $('#path_container ol');
  var $pathText = $('#path_container h3');
  var $contentOutterContainer = $('#content_outter_container');
  var $contentContainer = $('#content_container');
  var $contentContainerTitle = $('#content_container_title');
  var $pathSelectionContainer = $('#path_selection_container');
  var $entriesList = $('#path_selection_container #content_entries');
  var $resultViewer = $('#result_viewer');
  var $plotPlaceholder = $('#plot_placeholder');
  var $plotCreateButton = $('#content_outter_container form button');
  var $resetButton = $('#reset_button');
  var $metricsCombo = $('#metrics_combo');
  var $plotType = $('#plot_type');
  var $resultsLink = $('#results_link');
  var $plotModal = $('#plot_modal');

  var $evaluateTextArea = $('#evaluate_editor_txt');
  var $evaluateButton = $('#evaluate_button');
  var $evaluateResults = $('#evaluate_results');

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
          $pathSelectionContainer.show();
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
      setBreadcrumbs(groupid, expName, true);
      $resultViewer.jsonViewer(metrics, {collapsed: true});
      $contentOutterContainer.find('h5').text(groupid + '/' + expName);
      $contentOutterContainer.show();
      $pathSelectionContainer.hide();
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
      $pathSelectionContainer.show();
      loadGroup(groupid);
    });

    $newBreadcrumbGroupLink.appendTo($newBreadcrumbGroup);
    $newBreadcrumbGroup.appendTo($pathBreadcrumbs);

    var $newBreadcrumbExp = $breadcrumbTmpl.clone();
    var $newBreadcrumbExpLink = $breadcrumbLinkTmpl.clone();

    $newBreadcrumbExp.text(expName).appendTo($pathBreadcrumbs);
  };

  $resultsLink.click(function (e) {
    e.preventDefault();
    $pathSelectionContainer.show();
    loadGroups();
  });

  $evaluateButton.click(function (e) {
    e.preventDefault();

    currEvaluatorMappings = {
      m_opt: currMetrics['train_metrics_opt.json'],
      m_all: currMetrics['train_metrics_all.json'],
      m_val: currMetrics['validation_metrics.json']
    };

    try {    
      var evalText = $evaluateTextArea.val();
      var result = math.eval(evalText, currEvaluatorMappings);

      $evaluateResults.find('p').text(result);
    } catch (e) {
      $evaluateResults.find('p').text('ERROR: ' + e.toString());
    }
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
      'full_name': fullName,
      'time': new Date().getTime()
    };

    var plotUrl = generatePlotUrl + '?' + $.param(params);
    $plotPlaceholder.attr('src', plotUrl);
    $plotModal.modal('show');
  });
});