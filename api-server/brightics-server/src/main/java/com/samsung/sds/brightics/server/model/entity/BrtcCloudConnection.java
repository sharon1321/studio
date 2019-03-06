package com.samsung.sds.brightics.server.model.entity;

import lombok.Data;
import org.hibernate.validator.constraints.NotEmpty;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.io.Serializable;

@SuppressWarnings("serial")
@Data
@Entity
@Table(name="brtc_cloud_connection")
public class BrtcCloudConnection implements Serializable {

    @Id
    private String connectionName;

    private String cloudType;
    private String accessKeyId;
    private String secretAccessKey;

}
