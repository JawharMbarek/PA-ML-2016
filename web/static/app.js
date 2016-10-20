$(function () {
  var basePath = '/results';
  var resultsUrl = '/results';
  var groupUrl = '/results/:groupid';
  var expUrl = '/results/:groupid/:name';

  var $pathText = $('#path_container h3');
  var $entriesList = $('#content_container ul');
  var $loadingText = $('#content_container h3');
  var $resultViewer = $('#result_viewer');

  var loadExp = function (groupid, expName) {
    var currExpUrl = expUrl.replace(':groupid', groupid)
                           .replace(':name', expName);


    $.get(currExpUrl, function (metrics) {
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
});