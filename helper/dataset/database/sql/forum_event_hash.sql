(
  SELECT
  AccountUserHash as user_id,
  EventHash as forum_id,
  a.EventType as forum_type,
  a.TimeStamp as timestamp,
  c.PostType,
  c.PostTitle,
  c.PostText

  FROM project_himanshu.Request_2021Jan26_Forum_Events as a
  LEFT JOIN ca_{platform}.Forum_Events as b
  ON a.EventType = b.EventType  -- risky: we are assuming the timestamp and event_type are unique
  AND a.TimeStamp = b.TimeStamp
  AND a.DataPackageID = b.DataPackageID

  LEFT JOIN ca_{platform}.Forum_Info as c
  ON b.PostID = c.PostID
  AND b.DataPackageID = c.DataPackageID
  AND a.PostType = c.PostType
  WHERE a.DataPackageID in ('{course}')
)
UNION
(
  SELECT
  AccountUserHash as user_id,
  EventHash as forum_id,
  a.EventType as forum_type,
  a.TimeStamp as timestamp,
  c.PostType,
  c.PostTitle,
  c.PostText

  FROM project_himanshu.Request_2021Jun7_Forum_Events as a
  LEFT JOIN ca_{platform}.Forum_Events as b
  ON a.EventType = b.EventType
  -- risky: we are assuming the timestamp and event_type are unique
  AND a.TimeStamp = b.TimeStamp
  AND a.DataPackageID = b.DataPackageID

  LEFT JOIN ca_{platform}.Forum_Info as c
  ON b.PostID = c.PostID
  AND b.DataPackageID = c.DataPackageID
  AND a.PostType = c.PostType
  WHERE a.DataPackageID in ('{course}')
)
