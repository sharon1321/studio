package com.samsung.sds.brightics.server.model.entity.repository;

import com.samsung.sds.brightics.server.model.entity.BrtcCloudConnection;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.repository.query.Param;
import org.springframework.data.rest.core.annotation.RepositoryRestResource;

@RepositoryRestResource(path = "connection")
public interface BrtcCloudConnectionRepository extends BrtcRepository<BrtcCloudConnection, String> {
    Page<BrtcCloudConnection> findByCloudType(@Param("cloudType") String cloudType, Pageable pageable);
}
